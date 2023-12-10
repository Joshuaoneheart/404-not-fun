from mingpt.model import GPT
import numpy as np
from sklearn.cluster import KMeans
import json
from collections import deque
from gridworld import GridWorld
import torch
import torch.nn as nn
import random
from algorithm import SACAgent, GPTSACAgent
from tqdm import tqdm
import math
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
import metaworld
from random_process import OrnsteinUhlenbeckProcess
from memory import ReplayBuffer
import matplotlib.pyplot as plt

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
        x = np.array(x).reshape(-1, 39)
        y = x[1:]
        x = x[:-1]
        assert len(x) == len(y)
        return torch.FloatTensor(x).to(self.device), torch.FloatTensor(y).to(self.device)
device="cuda:0"
batch_size = 1
epoch_num = 100
num_workers = 0
lr = 5e-4
DISCOUNT_FACTOR=0.99
tau = 0.001
ou_theta=0.15
ou_mu=0.0
ou_sigma=0.2
update_frequency = 200
action_space = 4
train_episode_num = 1000
state_space = 39
n_tasks = 10
epsilon = 1
trajectory_num = 0
load_trajectory = False
max_steps = 1000
N = 100
method = "SAC"
gpt_model = "gpt-nano"
task_name = "reach-v2"
mt1 = metaworld.MT1(task_name)
env = mt1.train_classes[task_name]()
train_tasks = mt1.train_tasks[25:]
eval_tasks = mt1.train_tasks[:25]
trajectories = []
if load_trajectory:
    with open("trajectories.json") as fp:
        trajectories = json.load(fp)
    trajectory_num = 0
# collecting trajectory
agent = SACAgent(39, 4, [-1, 1], device, {"obs_dim": 39, "action_dim": 4, "hidden_dim": 1024, "hidden_depth": 3},
                 {"obs_dim": 39, "action_dim": 4, "hidden_dim": 1024, "hidden_depth": 3, "log_std_bounds": [-5, 2]}, DISCOUNT_FACTOR, 0.1, 1e-4, [0.9, 0.999],
                 1e-4, [0.9, 0.999], 1, 1e-4,
                 [0.9, 0.999], 0.005, 2, batch_size, True)
replay_buffer = ReplayBuffer((39,), (4, ), int(1e6), device)
step_prefix = 0
total_step = 0
x = []
y = []
for epoch in tqdm(range(0 if load_trajectory else train_episode_num)):
    x.append(step_prefix)
    env.set_task(random.choice(train_tasks))
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
    sr = 0
    for seed in range(len(eval_tasks)):
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
                break
            if done or info["success"]:
                sr += 1
                break
    y.append(sr / len(eval_tasks))
    print(f"mean step {y[-1]}")
plt.plot(x, smooth(y, 1), label="SAC")
trajectories_arr = []
for trajectory in trajectories:
    trajectories_arr.extend(trajectory)
kmeans = KMeans(n_clusters = 8192, random_state=0, n_init="auto").fit(np.array(trajectories_arr).astype('double'))
trajectories = trajectories[:100]
step_prefix = 0
for tr in trajectories:
    step_prefix += len(tr)
trajectories_arr = np.array(trajectories_arr)
train_dataset = TrajectoryDataset(trajectories, device, state_space)
model_config = GPT.get_default_config()
model_config.model_type = gpt_model
model_config.vocab_size = 8192
model_config.block_size = 501
model = GPT(model_config, action_space, state_space, n_tasks, kmeans)
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
train_loader = DataLoader(
            train_dataset,
            shuffle=True,
            batch_size=batch_size,
            num_workers=num_workers,
        )

print(model.kmeans.predict(np.array(trajectories[0])))

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
        self.not_done_no_max_buf = []
        self.ptr = 0
        self.size = buffer_size
        self.full = False

    def append(self, states, actions, rewards, not_dones, not_dones_no_max):
        self.state_buf.append(states)
        self.action_buf.append(actions)
        self.reward_buf.append(rewards)
        self.done_buf.append(not_dones)
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
        done_list = []
        not_done_no_max_list = []
        for idx in indexes:
            next_state_list.append(self.state_buf[idx][1:])
            state_list.append(self.state_buf[idx][:-1])
            action_list.append(self.action_buf[idx])
            reward_list.append(self.reward_buf[idx])
            done_list.append(self.done_buf[idx])
            not_done_no_max_list.append(self.not_done_no_max_buf[idx])

        return state_list, action_list, next_state_list, reward_list, done_list, not_done_no_max_list
    
    def __len__(self):
        if self.full:
            return self.size
        else:
            return self.ptr
buffer = EpisodeBuffer(1000, 1)
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
    env.set_task(random.choice(train_tasks))
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
        if len(buffer) >= 1:
            agent.update(buffer, total_step)
        total_step += 1
        current_state = next_state
    buffer.append(state_list, action_list, reward_list, not_done_list, not_done_no_max_list)
    step_prefix += step
    reward_record.append(np.mean(reward_list))
    steps = []
    sr = 0
    for seed in range(len(eval_tasks)):
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
                break
            if done or info["success"]:
                sr += 1
                print(f"{task_name} done in {step} steps")
                break
    y.append(sr / len(eval_tasks))
with open("kmeans_12.json", "w") as fp:
    json.dump({"x": x, "y": y}, fp)
plt.plot(x, smooth(y, 1), label="GPTSAC")
plt.legend()
plt.savefig(f"VQ.png")
