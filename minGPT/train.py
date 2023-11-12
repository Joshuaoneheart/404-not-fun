from mingpt.model import GPT
import numpy as np
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

class TrajectoryDataset(Dataset):
    def __init__(self, trajectories, device, state_space):
        self.x = trajectories
        self.device = device
    def __len__(self):
        return len(self.x)
    def __getitem__(self, idx):
        x = self.x[idx]
        y = x[1:]
        y.append(0)
        while len(x) < 1000:
            x.append(0)
            y.append(0)
        if len(x) > 1000:
            x = x[:1000]
            y = y[:1000]
        assert len(x) == 1000
        return torch.LongTensor(x).to(self.device), torch.LongTensor(y).to(self.device)
device="cuda:1"
batch_size = 32
epoch_num = 50
num_workers = 0
lr = 5e-4
update_frequency = 200
action_space = 4
train_episode_num = 500
state_space = 39
n_tasks = 10
max_steps = 1000
epsilon = 0.1
method = "GPT"
gpt_model = "gpt-nano"
mt10 = metaworld.MT10()
train_envs = []
for name, env_cls in mt10.train_classes.items():
    env = env_cls()
    task = random.choice([task for task in mt10.train_tasks
                        if task.env_name == name])
    env.set_task(task)
    train_envs.append(env)

trajectories = []
for env in train_envs:
    state, _ = env.reset()
    trajectory = [state]
    for step in range(max_steps):
        a = env.action_space.sample()
        state, reward, done, truncated, info = env.step(a)
        trajectory.append(state)
        if done or truncated:
            break
    trajectories.append(trajectory)
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
train_dataset = TrajectoryDataset(trajectories, device, state_space)
model_config = GPT.get_default_config()
model_config.model_type = gpt_model
model_config.vocab_size = 23
model_config.block_size = 1000
discretizer = QuantileDiscretizer(np.array(trajectories))
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
        logits, loss = model(x, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    print(f"Losses: {np.mean(losses)}")

class EpisodeBuffer:
    
    def __init__(self, buffer_size=1000, batch_size=32, max_steps=1000):
        self.sample_batch_size = batch_size
        self.state_buf = []
        self.action_buf = []
        self.reward_buf = []
        self.done_buf = []
        self.ptr = 0
        self.size = buffer_size
        self.full = False

    def append(self, states, actions, rewards, dones):
        self.state_buf.append(states)
        self.action_buf.append(actions)
        self.reward_buf.append(rewards)
        self.done_buf.append(dones)
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
        for idx in indexes:
            next_state_list.append(self.state_buf[idx][1:])
            state_list.append(self.state_buf[idx][:-1])
            action_list.append(self.action_buf[idx])
            reward_list.append(self.reward_buf[idx])
            done_list.append(self.done_buf[idx])

        return state_list, action_list, next_state_list, reward_list, done_list
    
    def __len__(self):
        if self.full:
            return self.size
        else:
            return self.ptr
buffer = EpisodeBuffer(1000, 1)
model_config = GPT.get_default_config()
model_config.model_type = gpt_model
model_config.vocab_size = 23
model_config.block_size = 1000 * state_space
actor = GPT(model_config, action_space, state_space, n_tasks,DDPG="A").to(device)
target_actor = GPT(model_config, action_space, state_space,n_tasks,DDPG="A").to(device)
critic = GPT(model_config, action_space, state_space, n_tasks, DDPG="C").to(device)
target_critic = GPT(model_config, action_space,state_space, n_tasks, DDPG="C").to(device)
actor.load_state_dict(model.state_dict())
target_actor.load_state_dict(model.state_dict())
critic.load_state_dict(model.state_dict())
target_critic.load_state_dict(model.state_dict())
total_step = 0
for episode in tqdm(range(train_episode_num)):
    for env in train_envs:
        current_state, _ = env.reset()
        state_list = [current_state]
        action_list = []
        reward_list = []
        done_list = []
        losses = []
        step = 0
        while step < max_steps:
            with torch.no_grad():
                logits, _ = actor(torch.LongTensor(state_list).unsqueeze(0).to(device))
            next_state, reward, is_done, info = env.step(action)
            state_list.append(next_state)
            action_list.append(action)
            reward_list.append(reward)
            done_list.append(is_done)
            if method == "DQN":
                losses.append(QL.learn(mem, total_step))
            step += 1
            if method == "DQN":
                total_step += 1
            if is_done:
                print(f"Done in {step} steps")
                break
            current_state = next_state
        buffer.append(state_list, action_list, reward_list, done_list)
        if len(buffer) >= 1 and method == "GPT":
            for batch in range(32):
                states, actions, next_states, rewards, dones = buffer.sample()
                states = torch.LongTensor(states).to(device)
                actions = torch.LongTensor(actions).to(device)
                state_action_values, _ = model(states)
                state_action_values=state_action_values.gather(2, actions.unsqueeze(2))
                next_states = torch.LongTensor(next_states).to(device)
                best_action, _ = model(next_states)
                best_action = best_action.argmax(2).unsqueeze(2)
                next_state_values, _ = target_model(next_states)
                next_state_values = next_state_values.gather(2, best_action)
                dones = 1 - torch.FloatTensor(dones).to(device)
                rewards = torch.FloatTensor(rewards).to(device)
                TDtargets = next_state_values.squeeze(2) * DISCOUNT_FACTOR * dones + rewards
                criterion = nn.MSELoss()
                
                loss = criterion(state_action_values.squeeze(2), TDtargets)
                losses.append(loss.item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_step += 1
                if total_step % update_frequency == 0:
                    target_model.load_state_dict(model.state_dict())
        run.log({"loss": np.mean(losses), "step": step})

run.finish()
