import numpy as np
import torch

class EpisodeBuffer:
    
    def __init__(self, buffer_size=1000, batch_size=32):
        self.sample_batch_size = batch_size
        self.state_buf = []
        self.action_buf = []
        self.reward_buf = []
        self.not_done_buf = []
        self.not_done_no_max_buf = []
        self.ptr = 0
        self.size = buffer_size
        self.full = False

    def append(self, states, actions, rewards, not_dones, not_dones_no_max):
        self.state_buf.append(states)
        self.action_buf.append(actions)
        self.reward_buf.append(rewards)
        self.not_done_buf.append(not_dones)
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
        for idx in indexes:
            next_state_list.append(self.state_buf[idx][1:])
            state_list.append(self.state_buf[idx][:-1])
            action_list.append(self.action_buf[idx])
            reward_list.append(self.reward_buf[idx])
            not_done_list.append(self.not_done_buf[idx])
            not_done_no_max_list.append(self.not_done_no_max_buf[idx])

        return state_list, action_list, next_state_list, reward_list, not_done_list, not_done_no_max_list
    
    def __len__(self):
        if self.full:
            return self.size
        else:
            return self.ptr


class ReplayBuffer(object):
    """Buffer to store environment transitions."""
    def __init__(self, obs_shape, action_shape, capacity, device):
        self.capacity = capacity
        self.device = device

        # the proprioceptive obs is stored as float32, pixels obs as uint8
        obs_dtype = np.float32 if len(obs_shape) == 1 else np.uint8

        self.obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.next_obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.actions = np.empty((capacity, *action_shape), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones_no_max = np.empty((capacity, 1), dtype=np.float32)

        self.idx = 0
        self.last_save = 0
        self.full = False

    def __len__(self):
        return self.capacity if self.full else self.idx

    def add(self, obs, action, reward, next_obs, done, done_no_max):
        np.copyto(self.obses[self.idx], obs)
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.next_obses[self.idx], next_obs)
        np.copyto(self.not_dones[self.idx], not done)
        np.copyto(self.not_dones_no_max[self.idx], not done_no_max)

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def sample(self, batch_size):
        idxs = np.random.randint(0,
                                 self.capacity if self.full else self.idx,
                                 size=batch_size)

        obses = torch.as_tensor(self.obses[idxs], device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        next_obses = torch.as_tensor(self.next_obses[idxs],
                                     device=self.device).float()
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)
        not_dones_no_max = torch.as_tensor(self.not_dones_no_max[idxs],
                                           device=self.device)

        return obses, actions, rewards, next_obses, not_dones, not_dones_no_max
