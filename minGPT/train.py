from mingpt.model import GPT2Model, GPT
from algorithm import SACAgent, GPTSACAgent
from transformers import GPT2Config
import numpy as np
import json
from collections import deque
from gridworld import GridWorld
import math
import torch
import torch.nn as nn
import random
import wandb
from tqdm import tqdm
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
import matplotlib.pyplot as plt
import metaworld
from memory import EpisodeBuffer, ReplayBuffer

def smooth(yValues, weight):
    smoothingWeight = min(math.sqrt(weight), 0.999)
    lastY = 0
    debiasWeight = 0
    rv = []
    for idx, yPoint in enumerate(yValues):
        lastY = lastY * smoothingWeight + yPoint
        debiasWeight = debiasWeight * smoothingWeight + 1
        rv.append(lastY / debiasWeight)
    return rv

class TrajectoryDataset(Dataset):
    def __init__(self, trajectories, device, state_space):
        self.x = trajectories
        self.device = device
    def __len__(self):
        return len(self.x)
    def __getitem__(self, idx):
        x = self.x[idx]
        x = np.array(x)
        y = x[1:]
        x = x[:-1]
        assert len(x) == len(y)
        return torch.FloatTensor(x).to(self.device), torch.FloatTensor(y).to(self.device)

device            = "cuda:0"
batch_size        = 1
epoch_num         = 1000
num_workers       = 0
lr                = 0.0001
DISCOUNT_FACTOR   = 0.99
tau               = 0.005
action_space      = 4
train_episode_num = 10000
state_space       = 39
n_tasks           = 10
epsilon           = 1
trajectory_num    = 10000
load_trajectory   = False
max_steps         = 1000
N                 = 1000
method            = "SAC"
gpt_model         = "gpt-nano"
task_name = "reach-v2"

trajectories = []
if load_trajectory:
    with open("trajectories.json") as fp:
        trajectories = json.load(fp)
    trajectory_num = 0
# collect trajectories
agent = SACAgent(39, 4, [-1, 1], device, {"obs_dim": 39, "action_dim": 4, "hidden_dim": 1024, "hidden_depth": 3},
                 {"obs_dim": 39, "action_dim": 4, "hidden_dim": 1024, "hidden_depth": 3, "log_std_bounds": [-5, 2]}, DISCOUNT_FACTOR, 0.1, 1e-4, [0.9, 0.999],
                 1e-4, [0.9, 0.999], 1, 1e-4,
                 [0.9, 0.999], 0.005, 2, batch_size, True)
total_step = 0
replay_buffer = ReplayBuffer((39,), (4, ), int(1e6), device)
mt1 = metaworld.MT1(task_name)
env = mt1.train_classes[task_name]()
eval_tasks = [random.choice(mt1.train_tasks) for _ in range(3)]
step_prefix = 0
x = []
y = []
for epoch in tqdm(range(train_episode_num)):
    x.append(step_prefix)
    env.set_task(random.choice(mt1.train_tasks))
    state, _ = env.reset(seed=42)
    trajectory = [list(state)]
    for step in range(max_steps):
        with torch.no_grad():
            action = agent.act(state, sample=True)
        next_state, reward, done, truncated, info = env.step(action)
        trajectory.append(list(state))
        done_no_max = 0 if truncated else done
        replay_buffer.add(state, action, reward, next_state, done, done_no_max)
        state = next_state
        if total_step > 32:
            agent.update(replay_buffer, total_step)
        total_step += 1
        if truncated:   
            step_prefix += step
            break
        if done or info["success"]:
            step_prefix += step
            print(f"{task_name} done in {step} steps")
            break
    trajectories.append(trajectory)
    steps = []
    for seed in range(3):
        env.set_task(eval_tasks[seed])
        state, _ = env.reset(seed=42)
        for step in range(max_steps):
            with torch.no_grad():
                action = agent.act(state, sample=True)
            next_state, reward, done, truncated, info = env.step(action)
            done_no_max = 0 if truncated else done
            state = next_state
            total_step += 1
            if truncated:   
                steps.append(step)
                break
            if done or info["success"]:
                steps.append(step)
                break
    y.append(np.mean(steps))
    print(f"mean step {y[-1]}")
plt.plot(x, smooth(y, 1), label="SAC")
'''
for seed in tqdm(range(trajectory_num // n_tasks)):
    for i, env in enumerate(train_envs):
        state, _ = env.reset(seed=seed)
        trajectory = [list(state)]
        for step in range(max_steps):
            a = np.random.uniform(low=-1, high=1, size=(4,))
            state, reward, done, truncated, info = env.step(a)
            trajectory.append(list(state))
            if truncated:
                break
            if done or info["success"]:
                break
        trajectories.append((i, trajectory))
        break
'''
if not load_trajectory:
    with open("trajectories.json", "w") as fp:
        json.dump(trajectories, fp)
print(f"data length: {len(trajectories)}")
# collecting trajectory
# trajectories = trajectories[:150]
step_prefix = 0
for tr in trajectories:
    step_prefix += len(tr)
print(step_prefix)
train_dataset = TrajectoryDataset(trajectories, device, state_space)
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
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
for epoch in tqdm(range(epoch_num)):
    losses = []
    for x, y in tqdm(train_loader):
        logits, loss = model(input_ids=x, targets=y)
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

buffer = EpisodeBuffer(10000, 1)
agent = GPTSACAgent(39, 4, [-1, 1], device, {"obs_dim": 39, "action_dim": 4, "hidden_dim": 1024, "hidden_depth": 3},
                 {"obs_dim": 39, "action_dim": 4, "hidden_dim": 1024, "hidden_depth": 3, "log_std_bounds": [-5, 2]}, DISCOUNT_FACTOR, 0.1, 1e-4, [0.9, 0.999],
                 1e-4, [0.9, 0.999], 1, 1e-4,
                 [0.9, 0.999], 0.005, 2,
                 batch_size, True, model)
total_step = 0
x = []
y = []
for episode in tqdm(range(train_episode_num)):
    x.append(step_prefix)
    critic_losses = []
    actor_losses = []
    reward_record = []
    env.seed(episode)
    env.set_task(random.choice(mt1.train_tasks))
    current_state, _ = env.reset()
    state_list = [current_state]
    action_list = []
    reward_list = []
    not_done_list = []
    not_done_no_max_list = []
    step = 0
    while step < max_steps:
        with torch.no_grad():
            action = agent.act(torch.FloatTensor(np.array(current_state)).to(device), sample=True)
        next_state, reward, is_done, truncated, info = env.step(action)
        state_list.append(next_state)
        action_list.append(action)
        reward_list.append(reward)
        not_done_list.append(not (is_done or info["success"]))
        not_done_no_max_list.append(not (0 if truncated else is_done))
        step += 1
        if is_done or info["success"]:
            print(f"{task_name} Done in {step} steps")
            break
        elif truncated:
            print(f"{task_name} truncated")
            break
        total_step += 1
        current_state = next_state
    buffer.append(state_list, action_list, reward_list, not_done_list, not_done_no_max_list)
    step_prefix += step
    reward_record.append(np.mean(reward_list))
    if len(buffer) >= 1 and method == "SAC":
        for batch in range(min(100, len(buffer))):
            agent.update(buffer, total_step)
    steps = []
    for seed in range(3):
        env.set_task(eval_tasks[seed])
        state, _ = env.reset(seed=42)
        for step in range(max_steps):
            with torch.no_grad():
                action = agent.act(state, sample=True)
            next_state, reward, done, truncated, info = env.step(action)
            done_no_max = 0 if truncated else done
            state = next_state
            total_step += 1
            if truncated:   
                steps.append(step)
                break
            if done or info["success"]:
                steps.append(step)
                print(f"{task_name} done in {step} steps")
                break
    y.append(np.mean(steps))
plt.plot(x, smooth(y, 1), label="GPTSAC")
plt.legend()
plt.savefig(f"SAC.png")
