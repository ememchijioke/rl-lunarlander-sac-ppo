# ppo/networks.py

import torch
import torch.nn as nn
from torch.distributions import Normal


LOG_STD_MIN = -20
LOG_STD_MAX = 2
EPSILON = 1e-6


def atanh(x):
    x = torch.clamp(x, -1.0 + EPSILON, 1.0 - EPSILON)
    return 0.5 * torch.log((1 + x) / (1 - x))


class ActorCritic(nn.Module):
    """
    Actor-Critic network for Custom Deep PPO.

    Actor:
        outputs a Gaussian policy for continuous actions.

    Critic:
        outputs V(s), the value estimate of the current state.
    """

    def __init__(self, obs_dim, action_dim, hidden_dim, action_limit):
        super().__init__()

        self.action_limit = action_limit

        self.shared_net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )

        self.mean_layer = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        self.value_layer = nn.Linear(hidden_dim, 1)

    def forward(self, obs):
        features = self.shared_net(obs)

        mean = self.mean_layer(features)
        log_std = torch.clamp(self.log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        value = self.value_layer(features)

        return mean, std, value

    def get_action_and_value(self, obs, action=None):
        """
        During rollout:
            action=None, so we sample a raw action, squash it with tanh,
            and return its corrected log probability.

        During update:
            action is the already-squashed environment action, so we invert it
            with atanh before recomputing the log probability.
        """

        mean, std, value = self.forward(obs)
        distribution = Normal(mean, std)

        if action is None:
            raw_action = distribution.rsample()
            squashed_action = torch.tanh(raw_action)
        else:
            squashed_action = action / self.action_limit
            raw_action = atanh(squashed_action)

        action_out = squashed_action * self.action_limit

        logprob = distribution.log_prob(raw_action)
        logprob -= torch.log(1 - squashed_action.pow(2) + EPSILON)
        logprob = logprob.sum(dim=-1, keepdim=True)

        entropy = distribution.entropy().sum(dim=-1, keepdim=True)

        return action_out, logprob, entropy, value

    def act_deterministic(self, obs):
        mean, _, _ = self.forward(obs)
        action = torch.tanh(mean) * self.action_limit
        return action