import numpy as np
import random
import operator
from typing import Callable, Dict, List
import torch

class ReplayMemory:
    """A simple numpy replay buffer."""

    def __init__(self, capacity):
        self.obs_buf = np.zeros([capacity, 22], dtype=np.float32)
        self.next_obs_buf = np.zeros([capacity, 22], dtype=np.float32)
        self.acts_buf = np.zeros([capacity], dtype=np.float32)
        self.rews_buf = np.zeros([capacity], dtype=np.float32)
        self.dones_buf = np.zeros([capacity], dtype=np.float32)
        self.max_size, self.batch_size = capacity, 32
        self.ptr, self.size, = 0, 0

    def append(
        self,
        obs: np.ndarray,
        act: np.ndarray, 
        next_obs: np.ndarray, 
        rew: float, 
        done
    ):
        self.obs_buf[self.ptr] = obs
        self.next_obs_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.dones_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size) -> Dict[str, np.ndarray]:
        idxs = np.random.choice(self.size, size=self.batch_size, replace=False)
        return None, self.obs_buf[idxs], self.acts_buf[idxs], self.next_obs_buf[idxs], self.rews_buf[idxs], self.dones_buf[idxs], None

