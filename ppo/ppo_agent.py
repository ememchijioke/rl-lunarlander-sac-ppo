

import os
import torch
import torch.nn.functional as F
from torch.optim import Adam

from ppo.networks import ActorCritic


class PPOAgent:
    """
    Custom Deep PPO agent for continuous control.

    This class contains:
    - action sampling
    - value prediction
    - PPO clipped policy update
    - critic/value loss
    - entropy bonus
    - save/load utilities
    """

    def __init__(
        self,
        obs_dim,
        action_dim,
        action_limit,
        hidden_dim,
        learning_rate,
        gamma,
        gae_lambda,
        clip_range,
        value_coef,
        entropy_coef,
        max_grad_norm,
        device,
    ):
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm

        self.network = ActorCritic(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            action_limit=action_limit,
        ).to(device)

        self.optimizer = Adam(self.network.parameters(), lr=learning_rate)

    def select_action(self, obs):
        """
        Sample an action during rollout collection.

        Returns:
        - action for the environment
        - log probability under the old policy
        - value estimate V(s)
        """

        obs_tensor = torch.as_tensor(
            obs,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        with torch.no_grad():
            action, logprob, _, value = self.network.get_action_and_value(obs_tensor)

        return (
            action.cpu().numpy()[0],
            logprob.cpu().numpy()[0],
            value.cpu().numpy()[0],
        )

    def get_value(self, obs):
        """
        Estimate V(s) for bootstrapping at the end of a rollout.
        """

        obs_tensor = torch.as_tensor(
            obs,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        with torch.no_grad():
            _, _, value = self.network.forward(obs_tensor)

        return value.cpu().numpy()[0, 0]

    def predict(self, obs, deterministic=True):
        """
        Select action for evaluation/video recording.

        Deterministic mode uses the policy mean.
        Non-deterministic mode samples from the policy.
        """

        obs_tensor = torch.as_tensor(
            obs,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        with torch.no_grad():
            if deterministic:
                action = self.network.act_deterministic(obs_tensor)
            else:
                action, _, _, _ = self.network.get_action_and_value(obs_tensor)

        return action.cpu().numpy()[0]

    def update(self, rollout_buffer, batch_size, update_epochs):
        """
        Run the PPO update.

        PPO uses the clipped surrogate objective:

        ratio = exp(new_logprob - old_logprob)

        actor_loss = -min(
            ratio * advantage,
            clipped_ratio * advantage
        )

        The critic is trained with MSE loss against computed returns.
        """

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy_loss = 0.0
        total_loss = 0.0
        update_count = 0

        for _ in range(update_epochs):
            for batch in rollout_buffer.get_batches(batch_size):
                obs = batch["obs"]
                actions = batch["actions"]
                old_logprobs = batch["old_logprobs"]
                advantages = batch["advantages"]
                returns = batch["returns"]

                _, new_logprobs, entropy, values = self.network.get_action_and_value(
                    obs,
                    action=actions,
                )

                ratio = torch.exp(new_logprobs - old_logprobs)

                unclipped_policy_loss = ratio * advantages
                clipped_policy_loss = (
                    torch.clamp(
                        ratio,
                        1.0 - self.clip_range,
                        1.0 + self.clip_range,
                    )
                    * advantages
                )

                policy_loss = -torch.min(
                    unclipped_policy_loss,
                    clipped_policy_loss,
                ).mean()

                value_loss = F.mse_loss(values, returns)

                entropy_loss = entropy.mean()

                loss = (
                    policy_loss
                    + self.value_coef * value_loss
                    - self.entropy_coef * entropy_loss
                )

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.network.parameters(),
                    self.max_grad_norm,
                )
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy_loss += entropy_loss.item()
                total_loss += loss.item()
                update_count += 1

        return {
            "policy_loss": total_policy_loss / update_count,
            "value_loss": total_value_loss / update_count,
            "entropy": total_entropy_loss / update_count,
            "total_loss": total_loss / update_count,
        }

    def save(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        torch.save(
            self.network.state_dict(),
            os.path.join(save_dir, "custom_ppo_actor_critic.pt"),
        )

    def load(self, save_dir):
        model_path = os.path.join(save_dir, "custom_ppo_actor_critic.pt")
        self.network.load_state_dict(
            torch.load(model_path, map_location=self.device)
        )
        self.network.eval()