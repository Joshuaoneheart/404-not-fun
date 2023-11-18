import numpy as np
from mingpt.model import GPT
import torch
import math
from torch import nn
import torch.nn.functional as F
from torch import distributions as pyd

def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)

def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data +
                                (1 - tau) * target_param.data)

def to_np(t):
    if t is None:
        return None
    elif t.nelement() == 0:
        return np.array([])
    else:
        return t.cpu().detach().numpy()

class TanhTransform(pyd.transforms.Transform):
    domain = pyd.constraints.real
    codomain = pyd.constraints.interval(-1.0, 1.0)
    bijective = True
    sign = +1

    def __init__(self, cache_size=1):
        super().__init__(cache_size=cache_size)

    @staticmethod
    def atanh(x):
        return 0.5 * (x.log1p() - (-x).log1p())

    def __eq__(self, other):
        return isinstance(other, TanhTransform)

    def _call(self, x):
        return x.tanh()

    def _inverse(self, y):
        # We do not clamp to the boundary here as it may degrade the performance of certain algorithms.
        # one should use `cache_size=1` instead
        return self.atanh(y)

    def log_abs_det_jacobian(self, x, y):
        # We use a formula that is more numerically stable, see details in the following link
        # https://github.com/tensorflow/probability/commit/ef6bb176e0ebd1cf6e25c6b5cecdd2428c22963f#diff-e120f70e92e6741bca649f04fcd907b7
        return 2. * (math.log(2.) - x - F.softplus(-2. * x))


class SquashedNormal(pyd.transformed_distribution.TransformedDistribution):
    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale

        self.base_dist = pyd.Normal(loc, scale)
        transforms = [TanhTransform()]
        super().__init__(self.base_dist, transforms)

    @property
    def mean(self):
        mu = self.loc
        for tr in self.transforms:
            mu = tr(mu)
        return mu


class DiagGaussianActor(nn.Module):
    """torch.distributions implementation of an diagonal Gaussian policy."""
    def __init__(self, config, gpt):
        super().__init__()
        obs_dim = config["obs_dim"]
        action_dim = config["action_dim"]
        hidden_dim = config["hidden_dim"]
        hidden_depth = config["hidden_depth"]
        log_std_bounds = config["log_std_bounds"]

        self.log_std_bounds = log_std_bounds
        model_config = GPT.get_default_config()
        model_config.model_type = 'gpt-nano'
        model_config.vocab_size = 1001
        model_config.block_size = 501

        self.trunk = GPT(model_config, 4, 39, 10,DDPG="A")
        self.trunk.load_state_dict(gpt.state_dict())

        self.outputs = dict()
        self.apply(weight_init)

    def forward(self, obs, task_id):
        mu, log_std = self.trunk(obs, task_id=task_id)[0].chunk(2, dim=-1)

        # constrain log_std inside [log_std_min, log_std_max]
        log_std = torch.tanh(log_std)
        log_std_min, log_std_max = self.log_std_bounds
        log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std +
                                                                     1)

        std = log_std.exp()

        self.outputs['mu'] = mu
        self.outputs['std'] = std

        dist = SquashedNormal(mu, std)
        return dist


class DoubleQCritic(nn.Module):
    """Critic network, employes double Q-learning."""
    def __init__(self, config, gpt):
        super().__init__()
        obs_dim = config["obs_dim"]
        action_dim = config["action_dim"]
        hidden_dim = config["hidden_dim"]
        hidden_depth = config["hidden_depth"]

        model_config = GPT.get_default_config()
        model_config.model_type = 'gpt-nano'
        model_config.vocab_size = 1001
        model_config.block_size = 501
        self.Q1 = GPT(model_config, 4, 39, 10, DDPG="C")
        self.Q1.load_state_dict(gpt.state_dict())
        model_config = GPT.get_default_config()
        model_config.model_type = 'gpt-nano'
        model_config.vocab_size = 1001
        model_config.block_size = 501
        self.Q2 = GPT(model_config, 4, 39, 10, DDPG="C")
        self.Q2.load_state_dict(gpt.state_dict())

        self.outputs = dict()
        self.apply(weight_init)

    def forward(self, obs, action, task_id):
        assert obs.size(0) == action.size(0)

        q1, _ = self.Q1(obs, action=action, task_id=task_id)
        q2, _ = self.Q2(obs, action=action, task_id=task_id)

        self.outputs['q1'] = q1
        self.outputs['q2'] = q2

        return q1, q2


class SACAgent:
    """SAC algorithm."""
    def __init__(self, obs_dim, action_dim, action_range, device, critic_cfg,
                 actor_cfg, discount, init_temperature, alpha_lr, alpha_betas,
                 actor_lr, actor_betas, actor_update_frequency, critic_lr,
                 critic_betas, critic_tau, critic_target_update_frequency,
                 batch_size, learnable_temperature, gpt):

        self.action_range = action_range
        self.device = torch.device(device)
        self.discount = discount
        self.critic_tau = critic_tau
        self.actor_update_frequency = actor_update_frequency
        self.critic_target_update_frequency = critic_target_update_frequency
        self.batch_size = batch_size
        self.learnable_temperature = learnable_temperature

        self.critic = DoubleQCritic(critic_cfg, gpt).to(self.device)
        self.critic_target = DoubleQCritic(critic_cfg, gpt).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor = DiagGaussianActor(actor_cfg, gpt).to(self.device)

        self.log_alpha = torch.tensor(np.log(init_temperature)).to(self.device)
        self.log_alpha.requires_grad = True
        # set target entropy to -|A|
        self.target_entropy = -action_dim

        # optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr,
                                                betas=actor_betas)

        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=critic_lr,
                                                 betas=critic_betas)

        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha],
                                                    lr=alpha_lr,
                                                    betas=alpha_betas)

        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def act(self, obs, task_id, sample=False):
        obs = torch.as_tensor(obs, dtype=torch.float, device=self.device)
        obs = obs.unsqueeze(0)
        dist = self.actor(obs, task_id=task_id)
        action = dist.sample() if sample else dist.mean
        action = action.clamp(*self.action_range)
        assert action.ndim == 2 and action.shape[0] == 1
        return to_np(action[0])

    def update_critic(self, obs, action, reward, next_obs, not_done, task_id, step):
        dist = self.actor(next_obs, task_id=task_id)
        next_action = dist.rsample()
        log_prob = dist.log_prob(next_action).sum(-1, keepdim=True)
        target_Q1, target_Q2 = self.critic_target(next_obs, next_action, task_id=task_id)
        target_V = torch.min(target_Q1,
                             target_Q2) - self.alpha.detach() * log_prob
        target_Q = reward.view(-1) + (not_done.view(-1) * self.discount * target_V.view(-1))
        target_Q = target_Q.detach()

        # get current Q estimates
        current_Q1, current_Q2 = self.critic(obs, action, task_id=task_id)
        critic_loss = F.mse_loss(current_Q1.view(-1), target_Q.view(-1)) + F.mse_loss(
            current_Q2.view(-1), target_Q.view(-1))

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

    def update_actor_and_alpha(self, obs, task_id, step):
        dist = self.actor(obs, task_id=task_id)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        actor_Q1, actor_Q2 = self.critic(obs, action, task_id=task_id)

        actor_Q = torch.min(actor_Q1, actor_Q2)
        actor_loss = (self.alpha.detach() * log_prob - actor_Q).mean()

        # optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        if self.learnable_temperature:
            self.log_alpha_optimizer.zero_grad()
            alpha_loss = (self.alpha *
                          (-log_prob - self.target_entropy).detach()).mean()
            alpha_loss.backward()
            self.log_alpha_optimizer.step()

    def update(self, buffer, step):
        task_ids, obs, action, next_obs, reward, not_done, not_done_no_max = buffer.sample()
        obs = torch.FloatTensor(np.array(obs)).to(self.device).view(-1, 39)
        action = torch.FloatTensor(np.array(action)).to(self.device).squeeze(0).view(-1, 4)
        next_obs = torch.FloatTensor(np.array(next_obs)).to(self.device).view(-1, 39)
        not_done = torch.FloatTensor(not_done).to(self.device).transpose(0, 1)
        not_done_no_max = torch.FloatTensor(not_done_no_max).to(self.device)
        reward = torch.FloatTensor(reward).to(self.device).transpose(0, 1)
        self.update_critic(obs, action, reward, next_obs, not_done_no_max, task_ids[0],step)

        if step % self.actor_update_frequency == 0:
            self.update_actor_and_alpha(obs, task_ids[0], step)

        if step % self.critic_target_update_frequency == 0:
            soft_update_params(self.critic, self.critic_target, self.critic_tau)
