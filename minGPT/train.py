from mingpt.model import GPT
import numpy as np
import json
from collections import deque
from gridworld import GridWorld
import torch
import torch.nn as nn
import random
import wandb
from tqdm import tqdm
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
import metaworld
from discretizer import QuantileDiscretizer
from random_process import OrnsteinUhlenbeckProcess

class TrajectoryDataset(Dataset):
    def __init__(self, trajectories, device, state_space, discretizer):
        self.x = trajectories
        self.device = device
        self.discretizer = discretizer
        self.m = state_space
    def __len__(self):
        return len(self.x)
    def __getitem__(self, idx):
        x = self.x[idx]
        x = self.discretizer.discretize(np.array(x)).reshape(-1)
        y = x[self.m:]
        x = x[:-self.m]
        assert len(x) == len(y)
        start = np.random.randint(0, max(1, len(x) // 39 - 25)) * 39
        x = x[start: start + 975]
        y = y[start: start + 975]
        return torch.LongTensor(x).to(self.device), torch.LongTensor(y).to(self.device)
device="cuda:1"
batch_size = 1
epoch_num = 500
num_workers = 0
lr = 5e-4
DISCOUNT_FACTOR=0.99
tau = 0.001
ou_theta=0.15
ou_mu=0.0
ou_sigma=0.2
update_frequency = 200
action_space = 4
train_episode_num = 500
state_space = 39
n_tasks = 10
epsilon = 1
trajectory_num = 0
load_trajectory = True
max_steps = 1000
N = 1000
method = "GPT"
gpt_model = "gpt-nano"
mt10 = metaworld.MT10()
train_envs = []
task_names = []
for name, env_cls in mt10.train_classes.items():
    env = env_cls()
    task = random.choice([task for task in mt10.train_tasks
                        if task.env_name == name])
    env.set_task(task)
    train_envs.append(env)
    task_names.append(name)

trajectories = []
if load_trajectory:
    with open("trajectories.json") as fp:
        trajectories = json.load(fp)
    trajectory_num = 0
for _ in range(trajectory_num // n_tasks):
    for env in train_envs:
        state, _ = env.reset()
        trajectory = [list(state)]
        for step in range(max_steps):
            a = env.action_space.sample()
            state, reward, done, truncated, info = env.step(a)
            trajectory.append(list(state))
            if done or truncated or info["success"]:
                break
    trajectories.append(trajectory)
if not load_trajectory:
    with open("trajectories.json", "w") as fp:
        json.dump(trajectories, fp)
run = wandb.init(project = "reinforcement learning final", config={
    "device": device,
    "method": method,
    "gpt_model": gpt_model,
    "epsilon": epsilon,
    "max_steps": max_steps,
    "epoch_num": epoch_num,
    "lr": lr,
    "train_episode_num": train_episode_num,
    "update_frequency": update_frequency,
    "batch_size": batch_size
        })
# collecting trajectory
model_config = GPT.get_default_config()
model_config.model_type = gpt_model
model_config.vocab_size = N + 1
model_config.block_size = 975
trajectories_arr = []
for trajectory in trajectories:
    trajectories_arr.extend(trajectory)
trajectories_arr = np.array(trajectories_arr)
discretizer = QuantileDiscretizer(trajectories_arr, N)
train_dataset = TrajectoryDataset(trajectories, device, state_space, discretizer)
random_process = OrnsteinUhlenbeckProcess(size=action_space, theta=ou_theta, mu=ou_mu, sigma=ou_sigma)
model = GPT(model_config, action_space, state_space, n_tasks)
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
train_loader = DataLoader(
            train_dataset,
            shuffle=True,
            batch_size=batch_size,
            num_workers=num_workers,
        )
for epoch in tqdm(range(epoch_num)):
    losses = []
    for x, y in tqdm(train_loader):
        logits, loss = model(x, targets=y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    print(f"Losses: {np.mean(losses)}")

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau
        )

class EpisodeBuffer:
    
    def __init__(self, buffer_size=1000, batch_size=32, max_steps=1000):
        self.sample_batch_size = batch_size
        self.state_buf = []
        self.action_buf = []
        self.reward_buf = []
        self.done_buf = []
        self.task_id_buf = []
        self.ptr = 0
        self.size = buffer_size
        self.full = False

    def append(self, task_id, states, actions, rewards, dones):
        self.state_buf.append(states)
        self.action_buf.append(actions)
        self.reward_buf.append(rewards)
        self.done_buf.append(dones)
        self.task_id_buf.append(task_id)
        self.ptr += 1
        if self.ptr == self.size:
            self.ptr = 0
            self.full = True

    def sample(self):
        indexes = np.random.choice(range(len(self)), min(self.sample_batch_size, len(self)), replace = False)
        state_list = []
        next_state_list = []
        action_list = []
        reward_list = []
        done_list = []
        task_id_list = []
        for idx in indexes:
            start = np.random.randint(0, max(1, len(self.done_buf[idx]) - 25))
            task_id_list.append(self.task_id_buf[idx])
            next_state_list.append(self.state_buf[idx][1:][start:start + 25])
            state_list.append(self.state_buf[idx][:-1][start:start + 25])
            action_list.append(self.action_buf[idx][start:start + 25])
            reward_list.append(self.reward_buf[idx][start:start + 25])
            done_list.append(self.done_buf[idx][start:start + 25])

        return task_id_list, state_list, action_list, next_state_list, reward_list, done_list
    
    def __len__(self):
        if self.full:
            return self.size
        else:
            return self.ptr
buffer = EpisodeBuffer(1000, 1)
model_config = GPT.get_default_config()
model_config.model_type = gpt_model
model_config.vocab_size = N + 1
model_config.block_size = 975
actor = GPT(model_config, action_space, state_space, n_tasks,DDPG="A").to(device)
model_config = GPT.get_default_config()
model_config.model_type = gpt_model
model_config.vocab_size = N + 1
model_config.block_size = 975
target_actor = GPT(model_config, action_space, state_space,n_tasks,DDPG="A").to(device)
model_config = GPT.get_default_config()
model_config.model_type = gpt_model
model_config.vocab_size = N + 1
model_config.block_size = 975
critic = GPT(model_config, action_space, state_space, n_tasks, DDPG="C").to(device)
model_config = GPT.get_default_config()
model_config.model_type = gpt_model
model_config.vocab_size = N + 1
model_config.block_size = 975
target_critic = GPT(model_config, action_space,state_space, n_tasks, DDPG="C").to(device)
actor.load_state_dict(model.state_dict())
target_actor.load_state_dict(model.state_dict())
critic.load_state_dict(model.state_dict())
target_critic.load_state_dict(model.state_dict())
actor_optimizer = torch.optim.AdamW(actor.parameters(), lr=lr)
critic_optimizer = torch.optim.AdamW(critic.parameters(), lr=lr)
total_step = 0
for episode in tqdm(range(train_episode_num)):
    critic_losses = []
    actor_losses = []
    steps = []
    reward_record = []
    for task_id, env in enumerate(train_envs):
        current_state, _ = env.reset()
        state_list = [current_state]
        action_list = []
        reward_list = []
        done_list = []
        step = 0
        while step < max_steps:
            with torch.no_grad():
                action, _ = actor(torch.LongTensor(np.array(state_list[-25:]).reshape(-1)).unsqueeze(0).to(device), task_id = task_id)
                action = action.cpu().data.numpy()[-1, :]
                action += epsilon * random_process.sample()
            next_state, reward, is_done, truncated, info = env.step(action)
            state_list.append(next_state)
            action_list.append(action)
            reward_list.append(reward)
            done_list.append(is_done or info["success"])
            step += 1
            if is_done or info["success"]:
                print(f"task {task_names[task_id]} Done in {step} steps")
                break
            elif truncated:
                print(f"task {task_names[task_id]} truncated")
                break
            current_state = next_state
        buffer.append(task_id, state_list, action_list, reward_list, done_list)
        steps.append(step)
        reward_record.append(np.mean(reward_list))
    epsilon = max(epsilon - 0.002, 0.05)
    if len(buffer) >= 1 and method == "GPT":
        for batch in range(32):
            task_ids, states, actions, next_states, rewards, dones = buffer.sample()
            states = torch.LongTensor(np.array(states)).view(1, -1).to(device)
            actions = torch.FloatTensor(actions).to(device).squeeze(0)
            state_action_values, _ = critic(states, action=actions, task_id=task_ids[0])
            next_states = torch.LongTensor(np.array(next_states)).view(1, -1).to(device)
            next_state_values, _ = target_critic(next_states, action=target_actor(next_states, task_id=task_ids[0])[0], task_id=task_ids[0])
            dones = 1 - torch.FloatTensor(dones).to(device)
            rewards = torch.FloatTensor(rewards).to(device)
            TDtargets = next_state_values.squeeze(1).unsqueeze(0) * DISCOUNT_FACTOR * dones + rewards
            criterion = nn.MSELoss()
            
            loss = criterion(state_action_values.squeeze(1).unsqueeze(0), TDtargets)
            critic_losses.append(loss.item())
            critic_optimizer.zero_grad()
            loss.backward()
            critic_optimizer.step()
            actor_optimizer.zero_grad()
            policy_loss = - critic(states, action=actor(states, task_id=task_ids[0])[0], task_id=task_ids[0])[0]
            actor_losses.append(policy_loss.mean().item())
            policy_loss.mean().backward()
            actor_optimizer.step()
            total_step += 1
            soft_update(target_actor, actor, tau)
            soft_update(target_critic, critic, tau)

    run.log({"actor_loss": np.mean(actor_losses), "critic_loss": np.mean(critic_losses), "step": np.mean(steps), "avg_reward": np.mean(reward_record), "epsilon": epsilon})

run.finish()
