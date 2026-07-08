
import numpy as np
import torch


class RolloutBuffer:
    """
    Rollout buffer for Custom Deep PPO.

    PPO is an on-policy algorithm, so it does not use an experience replay buffer
    like DQN or SAC. It rather collects a fixed number of fresh environment
    transitions, computes returns and advantages, updates the policy, and then
    clears the buffer.

    This buffer is designed for continuous-control environments such as
    LunarLanderContinuous-v3, where actions are real-valued vectors.
    """

    def __init__(self, obs_dim, action_dim, size, device):
        self.size = size
        self.ptr = 0
        self.device = device

        # Observation/state from the environment.
        self.obs_buf = np.zeros((size, obs_dim), dtype=np.float32)

        # Continuous action vector sampled from the policy.
        self.action_buf = np.zeros((size, action_dim), dtype=np.float32)

        # Log probability of the action under the old policy.
        # PPO uses this to compute the policy probability ratio:
        # ratio = exp(new_logprob - old_logprob)
        self.logprob_buf = np.zeros((size, 1), dtype=np.float32)

        # Reward received after taking the action.
        self.reward_buf = np.zeros(size, dtype=np.float32)

        # Done flag showing whether the episode ended.
        self.done_buf = np.zeros(size, dtype=np.float32)

        # Value estimate V(s) predicted by the critic at collection time.
        self.value_buf = np.zeros(size, dtype=np.float32)

        # Advantage estimates used to update the policy.
        self.advantage_buf = np.zeros(size, dtype=np.float32)

        # Discounted returns used as the target for value-function learning.
        self.return_buf = np.zeros(size, dtype=np.float32)

    def store(self, obs, action, logprob, reward, done, value):
        """
        Store one transition collected from the environment.

        Each transition contains:
        state, action, old log probability, reward, done flag, and value estimate.
        """

        if self.ptr >= self.size:
            raise RuntimeError(
                "RolloutBuffer is full. Call clear() before storing more data."
            )

        self.obs_buf[self.ptr] = obs
        self.action_buf[self.ptr] = action
        self.logprob_buf[self.ptr] = logprob
        self.reward_buf[self.ptr] = reward
        self.done_buf[self.ptr] = done
        self.value_buf[self.ptr] = value

        self.ptr += 1

    def compute_returns_and_advantages(self, last_value, gamma, gae_lambda):
        """
        Compute Generalized Advantage Estimation.

        """

        advantage = 0.0

        for step in reversed(range(self.ptr)):
            if step == self.ptr - 1:
                next_value = last_value
            else:
                next_value = self.value_buf[step + 1]

            # If done = 1, the next state is terminal, so no future value is used.
            next_non_terminal = 1.0 - self.done_buf[step]

            delta = (
                self.reward_buf[step]
                + gamma * next_value * next_non_terminal
                - self.value_buf[step]
            )

            advantage = delta + gamma * gae_lambda * next_non_terminal * advantage
            self.advantage_buf[step] = advantage

        # Return target for critic training.
        self.return_buf[:self.ptr] = (
            self.advantage_buf[:self.ptr] + self.value_buf[:self.ptr]
        )

    def get_batches(self, batch_size):
        """
        Yield shuffled minibatches for PPO updates.

        PPO usually updates the policy for several epochs over the same rollout.
        Advantage normalization improves training stability.
        """

        indices = np.arange(self.ptr)
        np.random.shuffle(indices)

        advantages = self.advantage_buf[:self.ptr]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for start in range(0, self.ptr, batch_size):
            end = start + batch_size
            batch_idx = indices[start:end]

            yield {
                "obs": torch.as_tensor(
                    self.obs_buf[batch_idx],
                    dtype=torch.float32,
                    device=self.device,
                ),
                "actions": torch.as_tensor(
                    self.action_buf[batch_idx],
                    dtype=torch.float32,
                    device=self.device,
                ),
                "old_logprobs": torch.as_tensor(
                    self.logprob_buf[batch_idx],
                    dtype=torch.float32,
                    device=self.device,
                ),
                "advantages": torch.as_tensor(
                    advantages[batch_idx],
                    dtype=torch.float32,
                    device=self.device,
                ).unsqueeze(-1),
                "returns": torch.as_tensor(
                    self.return_buf[batch_idx],
                    dtype=torch.float32,
                    device=self.device,
                ).unsqueeze(-1),
                "old_values": torch.as_tensor(
                    self.value_buf[batch_idx],
                    dtype=torch.float32,
                    device=self.device,
                ).unsqueeze(-1),
            }

    def clear(self):
        """
        Clear the buffer after PPO finishes updating from the collected rollout.
        The stored arrays are reused, but the pointer is reset.
        """

        self.ptr = 0