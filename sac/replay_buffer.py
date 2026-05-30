# sac/replay_buffer.py

import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, obs_dim, action_dim, size, device):
        self.obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.next_obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.action_buf = np.zeros((size, action_dim), dtype=np.float32)
        self.reward_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)

        self.max_size = size
        self.ptr = 0
        self.size = 0
        self.device = device

    def store(self, obs, action, reward, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.action_buf[self.ptr] = action
        self.reward_buf[self.ptr] = reward
        self.next_obs_buf[self.ptr] = next_obs
        self.done_buf[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size):
        idxs = np.random.randint(0, self.size, size=batch_size)

        batch = {
            "obs": torch.as_tensor(self.obs_buf[idxs], device=self.device),
            "actions": torch.as_tensor(self.action_buf[idxs], device=self.device),
            "rewards": torch.as_tensor(self.reward_buf[idxs], device=self.device).unsqueeze(-1),
            "next_obs": torch.as_tensor(self.next_obs_buf[idxs], device=self.device),
            "dones": torch.as_tensor(self.done_buf[idxs], device=self.device).unsqueeze(-1),
        }

        return batch