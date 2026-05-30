
import os
import torch
import torch.nn.functional as F
from torch.optim import Adam

from sac.networks import Actor, Critic


class SACAgent:
    def __init__(
        self,
        obs_dim,
        action_dim,
        action_limit,
        hidden_dim,
        actor_lr,
        critic_lr,
        alpha_lr,
        gamma,
        tau,
        device,
    ):
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.action_limit = action_limit

        self.actor = Actor(obs_dim, action_dim, hidden_dim, action_limit).to(device)

        self.critic1 = Critic(obs_dim, action_dim, hidden_dim).to(device)
        self.critic2 = Critic(obs_dim, action_dim, hidden_dim).to(device)

        self.target_critic1 = Critic(obs_dim, action_dim, hidden_dim).to(device)
        self.target_critic2 = Critic(obs_dim, action_dim, hidden_dim).to(device)

        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())

        self.actor_optimizer = Adam(self.actor.parameters(), lr=actor_lr)
        self.critic1_optimizer = Adam(self.critic1.parameters(), lr=critic_lr)
        self.critic2_optimizer = Adam(self.critic2.parameters(), lr=critic_lr)

        self.target_entropy = -float(action_dim)
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha_optimizer = Adam([self.log_alpha], lr=alpha_lr)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def select_action(self, obs, deterministic=False):
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            action, _, deterministic_action = self.actor.sample(obs_tensor)

        if deterministic:
            return deterministic_action.cpu().numpy()[0]

        return action.cpu().numpy()[0]

    def update(self, replay_buffer, batch_size):
        batch = replay_buffer.sample_batch(batch_size)

        obs = batch["obs"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        next_obs = batch["next_obs"]
        dones = batch["dones"]

        with torch.no_grad():
            next_actions, next_log_probs, _ = self.actor.sample(next_obs)

            target_q1 = self.target_critic1(next_obs, next_actions)
            target_q2 = self.target_critic2(next_obs, next_actions)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_probs

            backup = rewards + self.gamma * (1.0 - dones) * target_q

        current_q1 = self.critic1(obs, actions)
        current_q2 = self.critic2(obs, actions)

        critic1_loss = F.mse_loss(current_q1, backup)
        critic2_loss = F.mse_loss(current_q2, backup)

        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        new_actions, log_probs, _ = self.actor.sample(obs)

        q1_new = self.critic1(obs, new_actions)
        q2_new = self.critic2(obs, new_actions)
        q_new = torch.min(q1_new, q2_new)

        actor_loss = (self.alpha.detach() * log_probs - q_new).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        self.soft_update(self.critic1, self.target_critic1)
        self.soft_update(self.critic2, self.target_critic2)

        return {
            "critic1_loss": critic1_loss.item(),
            "critic2_loss": critic2_loss.item(),
            "actor_loss": actor_loss.item(),
            "alpha_loss": alpha_loss.item(),
            "alpha": self.alpha.item(),
        }

    def soft_update(self, source_net, target_net):
        for source_param, target_param in zip(source_net.parameters(), target_net.parameters()):
            target_param.data.copy_(
                self.tau * source_param.data + (1.0 - self.tau) * target_param.data
            )

    def save(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)

        torch.save(self.actor.state_dict(), os.path.join(save_dir, "actor.pt"))
        torch.save(self.critic1.state_dict(), os.path.join(save_dir, "critic1.pt"))
        torch.save(self.critic2.state_dict(), os.path.join(save_dir, "critic2.pt"))
        torch.save(self.log_alpha.detach().cpu(), os.path.join(save_dir, "log_alpha.pt"))

    def load(self, save_dir):
        self.actor.load_state_dict(
            torch.load(os.path.join(save_dir, "actor.pt"), map_location=self.device)
        )
        self.critic1.load_state_dict(
            torch.load(os.path.join(save_dir, "critic1.pt"), map_location=self.device)
        )
        self.critic2.load_state_dict(
            torch.load(os.path.join(save_dir, "critic2.pt"), map_location=self.device)
        )

        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())

        loaded_log_alpha = torch.load(
            os.path.join(save_dir, "log_alpha.pt"),
            map_location=self.device,
        )
        self.log_alpha.data.copy_(loaded_log_alpha.to(self.device))