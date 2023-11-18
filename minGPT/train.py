from mingpt.model import GPT2Model, GPT
from algorithm import SACAgent
from transformers import GPT2Config
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
from random_process import OrnsteinUhlenbeckProcess
from metaworld.policies import SawyerReachV2Policy,SawyerPushV2Policy,SawyerPickPlaceV2Policy,SawyerDoorOpenV2Policy,SawyerDrawerOpenV2Policy,SawyerDrawerCloseV2Policy,SawyerButtonPressTopdownV2Policy,SawyerPegInsertionSideV2Policy,SawyerWindowOpenV2Policy,SawyerWindowCloseV2Policy

class TrajectoryDataset(Dataset):
    def __init__(self, trajectories, device, state_space):
        self.x = trajectories
        self.device = device
    def __len__(self):
        return len(self.x)
    def __getitem__(self, idx):
        task_id, x = self.x[idx]
        x = np.array(x)
        y = x
        x = x[:-1]
        assert len(x) + 1 == len(y)
        return task_id, torch.FloatTensor(x).to(self.device), torch.FloatTensor(y).to(self.device)
device            = "cuda:0"
batch_size        = 1
epoch_num         = 100
num_workers       = 0
lr                = 0.00001
DISCOUNT_FACTOR   = 0.99
tau               = 0.005
action_space      = 4
train_episode_num = 500
state_space       = 39
n_tasks           = 10
epsilon           = 1
trajectory_num    = 10000
load_trajectory   = False
max_steps         = 1000
N                 = 1000
method            = "SAC"
gpt_model         = "gpt-nano"
mt10 = metaworld.MT10()
task_policies = {
        "reach-v2": SawyerReachV2Policy(),
        "push-v2": SawyerPushV2Policy(),
        "pick-place-v2": SawyerPickPlaceV2Policy(),
        "door-open-v2": SawyerDoorOpenV2Policy(),
        "drawer-open-v2":SawyerDrawerOpenV2Policy(),
        "drawer-close-v2":SawyerDrawerCloseV2Policy(),
        "button-press-topdown-v2": SawyerButtonPressTopdownV2Policy(),
        "peg-insert-side-v2": SawyerPegInsertionSideV2Policy(),
        "window-open-v2": SawyerWindowOpenV2Policy(),
        "window-close-v2": SawyerWindowCloseV2Policy()
}
train_envs = []
task_names = []
for name, env_cls in mt10.train_classes.items():
    env = env_cls()
    task = random.choice([task for task in mt10.train_tasks
                        if task.env_name == name])
    env.set_task(task)
    train_envs.append(env)
    task_names.append(name)
    break

trajectories = []
if load_trajectory:
    with open("trajectories.json") as fp:
        trajectories = json.load(fp)
    trajectory_num = 0
for seed in tqdm(range(trajectory_num // n_tasks)):
    mt10 = metaworld.MT10(seed=seed)
    for i, (name, env_cls) in enumerate(mt10.train_classes.items()):
        env = env_cls()
        task = random.choice([task for task in mt10.train_tasks
                            if task.env_name == name])
        policy = task_policies[name]
        env.set_task(task)
        state, _ = env.reset()
        trajectory = [list(state)]
        for step in range(max_steps):
            a = policy.get_action(state)
            state, reward, done, truncated, info = env.step(a)
            trajectory.append(list(state))
            if truncated:
                break
            if done or info["success"]:
                print(f"{name} done in {step} steps")
                break
        trajectories.append((i, trajectory))
        break

for seed in tqdm(range(trajectory_num // n_tasks)):
    mt10 = metaworld.MT10(seed=seed)
    for i, (name, env_cls) in enumerate(mt10.train_classes.items()):
        env = env_cls()
        task = random.choice([task for task in mt10.train_tasks
                            if task.env_name == name])
        env.set_task(task)
        state, _ = env.reset()
        trajectory = [list(state)]
        for step in range(max_steps):
            a = env.action_space.sample()
            state, reward, done, truncated, info = env.step(a)
            trajectory.append(list(state))
            if truncated:
                break
            if done or info["success"]:
                break
        trajectories.append((i, trajectory))
        break
if not load_trajectory:
    with open("trajectories.json", "w") as fp:
        json.dump(trajectories, fp)
print(f"data length: {len(trajectories)}")
run = wandb.init(project = "reinforcement learning final", config={
    "device": device,
    "method": method,
    "gpt_model": gpt_model,
    "epsilon": epsilon,
    "max_steps": max_steps,
    "epoch_num": epoch_num,
    "lr": lr,
    "train_episode_num": train_episode_num,
    "batch_size": batch_size
        })
# collecting trajectory
train_dataset = TrajectoryDataset(trajectories, device, state_space)
random_process = OrnsteinUhlenbeckProcess(size=action_space, theta=ou_theta, mu=ou_mu, sigma=ou_sigma)
# config = GPT2Config()
model_config = GPT.get_default_config()
model_config.model_type = gpt_model
model_config.vocab_size = N + 1
model_config.block_size = 501
model = GPT(model_config, action_space, state_space, n_tasks)
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
train_loader = DataLoader(
            train_dataset,
            shuffle=True,
            batch_size=batch_size,
            num_workers=num_workers,
        )
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
for epoch in tqdm(range(epoch_num)):
    losses = []
    for task_id, x, y in tqdm(train_loader):
        logits, loss = model(input_ids=x, targets=y, task_id=task_id)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    scheduler.step()
    print(scheduler.get_last_lr())
    print(f"Losses: {np.mean(losses)}")

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau
        )

class EpisodeBuffer:
    
    def __init__(self, buffer_size=1000, batch_size=32):
        self.sample_batch_size = batch_size
        self.state_buf = []
        self.action_buf = []
        self.reward_buf = []
        self.not_done_buf = []
        self.not_done_no_max_buf = []
        self.task_id_buf = []
        self.ptr = 0
        self.size = buffer_size
        self.full = False

    def append(self, task_id, states, actions, rewards, not_dones, not_dones_no_max):
        self.state_buf.append(states)
        self.action_buf.append(actions)
        self.reward_buf.append(rewards)
        self.not_done_buf.append(not_dones)
        self.task_id_buf.append(task_id)
        self.not_done_no_max_buf.append(not_dones_no_max)
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
        not_done_list = []
        not_done_no_max_list = []
        task_id_list = []
        for idx in indexes:
            task_id_list.append(self.task_id_buf[idx])
            next_state_list.append(self.state_buf[idx][1:])
            state_list.append(self.state_buf[idx][:-1])
            action_list.append(self.action_buf[idx])
            reward_list.append(self.reward_buf[idx])
            not_done_list.append(self.not_done_buf[idx])
            not_done_no_max_list.append(self.not_done_no_max_buf[idx])

        return task_id_list, state_list, action_list, next_state_list, reward_list, not_done_list, not_done_no_max_list
    
    def __len__(self):
        if self.full:
            return self.size
        else:
            return self.ptr
buffer = EpisodeBuffer(10000, 1)
agent = SACAgent(39, 4, [-1, 1], device, {"obs_dim": 39, "action_dim": 4, "hidden_dim": 1024, "hidden_depth": 3},
                 {"obs_dim": 39, "action_dim": 4, "hidden_dim": 1024, "hidden_depth": 3, "log_std_bounds": [-5, 2]}, DISCOUNT_FACTOR, 0.1, 1e-4, [0.9, 0.999],
                 1e-4, [0.9, 0.999], 1, 1e-4,
                 [0.9, 0.999], 0.005, 2,
                 batch_size, True, model)
total_step = 0
for episode in tqdm(range(train_episode_num)):
    critic_losses = []
    actor_losses = []
    steps = []
    reward_record = []
    for task_id, env in enumerate(train_envs):
        env.seed(episode)
        current_state, _ = env.reset()
        state_list = [current_state]
        action_list = []
        reward_list = []
        not_done_list = []
        not_done_no_max_list = []
        step = 0
        while step < max_steps:
            with torch.no_grad():
                action = agent.act(torch.FloatTensor(np.array(current_state)).to(device), task_id = task_id, sample=True)
            next_state, reward, is_done, truncated, info = env.step(action)
            state_list.append(next_state)
            action_list.append(action)
            reward_list.append(reward)
            not_done_list.append(not (is_done or info["success"]))
            not_done_no_max_list.append(not (0 if truncated else is_done))
            step += 1
            if is_done or info["success"]:
                print(f"task {task_names[task_id]} Done in {step} steps")
                break
            elif truncated:
                print(f"task {task_names[task_id]} truncated")
                break
            total_step += 1
            current_state = next_state
        buffer.append(task_id, state_list, action_list, reward_list, not_done_list, not_done_no_max_list)
        steps.append(step)
        reward_record.append(np.mean(reward_list))
    if len(buffer) >= 1 and method == "SAC":
        for batch in range(min(100, len(buffer))):
            agent.update(buffer, total_step)
    if len(buffer) >= 1 and method == "GPT":
        for batch in range(100):
            task_ids, states, actions, next_states, rewards, dones = buffer.sample()
            states = torch.FloatTensor(np.array(states)).to(device)
            actions = torch.FloatTensor(np.array(actions)).to(device).squeeze(0)
            state_action_values, _ = critic(input_ids=states, action=actions, task_id=task_ids[0])
            next_states = torch.FloatTensor(np.array(next_states)).to(device)
            with torch.no_grad():
                next_state_values, _ = target_critic(input_ids=next_states, action=target_actor(next_states, task_id=task_ids[0])[0], task_id=task_ids[0])
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
            policy_loss = - critic(input_ids=states, action=actor(input_ids=states, task_id=task_ids[0])[0], task_id=task_ids[0])[0]
            actor_losses.append(policy_loss.mean().item())
            policy_loss.mean().backward()
            actor_optimizer.step()
            total_step += 1
        soft_update(target_actor, actor, tau)
        soft_update(target_critic, critic, tau)
    run.log({"step": np.mean(steps), "avg_reward": np.mean(reward_record), "epsilon": epsilon})

run.finish()
