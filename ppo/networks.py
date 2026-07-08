# ppo/networks.py

import torch
import torch.nn as nn
from torch.distributions import Normal


LOG_STD_MIN = -20
LOG_STD_MAX = 2
EPSILON = 1e-6


class ActorCritic(nn.Module):
    """
    Actor-Critic network for Custom Deep PPO.

    The actor outputs a Gaussian policy for continuous actions.
    The critic outputs the value estimate V(s).
    """

    def __init__(self, obs_dim, action_dim, hidden_dim, action_limit):
        super().__init__()

        self.action_limit = action_limit

        # Shared feature extractor
        self.shared_net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )

        # Actor head: outputs mean action
        self.mean_layer = nn.Linear(hidden_dim, action_dim)

        # Learnable log standard deviation for exploration
        self.log_std = nn.Parameter(torch.zeros(action_dim))

        # Critic head: outputs state value V(s)
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
        Sample or evaluate an action.

        During rollout collection:
            action=None, so the policy samples an action.

        During PPO update:
            action is provided, so the method recomputes log probability
            under the current policy.
        """

        mean, std, value = self.forward(obs)
        distribution = Normal(mean, std)

        if action is None:
            raw_action = distribution.rsample()
        else:
            raw_action = action

        logprob = distribution.log_prob(raw_action).sum(dim=-1, keepdim=True)
        entropy = distribution.entropy().sum(dim=-1, keepdim=True)

        # Squash action to environment bounds
        squashed_action = torch.tanh(raw_action) * self.action_limit

        return squashed_action, logprob, entropy, value

    def act_deterministic(self, obs):
        """
        Deterministic action for evaluation.
        Uses the policy mean instead of sampling.
        """

        mean, _, _ = self.forward(obs)
        action = torch.tanh(mean) * self.action_limit
        return action