# sac/networks.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


LOG_STD_MIN = -20
LOG_STD_MAX = 2
EPSILON = 1e-6


class Actor(nn.Module):
    """
    Gaussian policy network for SAC.

    Input:
        observation

    Output:
        continuous action sampled using reparameterization trick
    """

    def __init__(self, obs_dim, action_dim, hidden_dim, action_limit):
        super().__init__()

        self.action_limit = action_limit

        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.mean_layer = nn.Linear(hidden_dim, action_dim)
        self.log_std_layer = nn.Linear(hidden_dim, action_dim)

    def forward(self, obs):
        x = self.net(obs)

        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)

        return mean, log_std

    def sample(self, obs):
        mean, log_std = self.forward(obs)
        std = log_std.exp()

        normal = Normal(mean, std)
        z = normal.rsample()

        squashed_action = torch.tanh(z)
        action = squashed_action * self.action_limit

        log_prob = normal.log_prob(z)
        log_prob -= torch.log(1 - squashed_action.pow(2) + EPSILON)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        deterministic_action = torch.tanh(mean) * self.action_limit

        return action, log_prob, deterministic_action


class Critic(nn.Module):
    """
    Q-value network.

    Input:
        observation + action

    Output:
        Q-value estimate
    """

    def __init__(self, obs_dim, action_dim, hidden_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs, action):
        x = torch.cat([obs, action], dim=-1)
        q_value = self.net(x)
        return q_value