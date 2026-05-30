# train_custom_sac.py

import csv
import os
import random
from collections import deque

import numpy as np
import torch
from tqdm import tqdm

from config import (
    ENV_NAME,
    SEED,
    SAC_TOTAL_STEPS,
    SAC_MODEL_DIR,
    SAC_LOG_DIR,
    GAMMA,
    TAU,
    ACTOR_LR,
    CRITIC_LR,
    ALPHA_LR,
    BUFFER_SIZE,
    BATCH_SIZE,
    HIDDEN_DIM,
    START_STEPS,
    UPDATE_AFTER,
    UPDATE_EVERY,
    MAX_EPISODE_STEPS,
)
from env_utils import make_env
from sac.replay_buffer import ReplayBuffer
from sac.sac_agent import SACAgent


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def write_log_header(log_path):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    with open(log_path, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "episode",
                "total_steps",
                "episode_reward",
                "episode_length",
                "avg_reward_10",
                "critic1_loss",
                "critic2_loss",
                "actor_loss",
                "alpha_loss",
                "alpha",
            ]
        )


def append_log(log_path, row):
    with open(log_path, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(row)


def main():
    set_seed(SEED)

    os.makedirs(SAC_MODEL_DIR, exist_ok=True)
    os.makedirs(SAC_LOG_DIR, exist_ok=True)

    log_path = os.path.join(SAC_LOG_DIR, "training_log.csv")
    write_log_header(log_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    env = make_env(ENV_NAME, render_mode=None, seed=SEED)

    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_limit = float(env.action_space.high[0])

    print(f"Environment: {ENV_NAME}")
    print(f"Observation dim: {obs_dim}")
    print(f"Action dim: {action_dim}")
    print(f"Action limit: {action_limit}")

    replay_buffer = ReplayBuffer(
        obs_dim=obs_dim,
        action_dim=action_dim,
        size=BUFFER_SIZE,
        device=device,
    )

    agent = SACAgent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        action_limit=action_limit,
        hidden_dim=HIDDEN_DIM,
        actor_lr=ACTOR_LR,
        critic_lr=CRITIC_LR,
        alpha_lr=ALPHA_LR,
        gamma=GAMMA,
        tau=TAU,
        device=device,
    )

    obs, info = env.reset(seed=SEED)

    episode_reward = 0.0
    episode_length = 0
    episode = 1

    recent_rewards = deque(maxlen=10)

    latest_losses = {
        "critic1_loss": 0.0,
        "critic2_loss": 0.0,
        "actor_loss": 0.0,
        "alpha_loss": 0.0,
        "alpha": 0.0,
    }

    progress_bar = tqdm(range(1, SAC_TOTAL_STEPS + 1), desc="Training Custom SAC")

    for total_step in progress_bar:
        if total_step < START_STEPS:
            action = env.action_space.sample()
        else:
            action = agent.select_action(obs, deterministic=False)

        next_obs, reward, terminated, truncated, info = env.step(action)

        episode_length += 1
        episode_reward += reward

        timeout = episode_length >= MAX_EPISODE_STEPS
        done = terminated or truncated or timeout

        replay_buffer.store(
            obs=obs,
            action=action,
            reward=reward,
            next_obs=next_obs,
            done=float(terminated or truncated),
        )

        obs = next_obs

        if total_step >= UPDATE_AFTER and replay_buffer.size >= BATCH_SIZE:
            if total_step % UPDATE_EVERY == 0:
                latest_losses = agent.update(replay_buffer, BATCH_SIZE)

        if done:
            recent_rewards.append(episode_reward)
            avg_reward_10 = float(np.mean(recent_rewards))

            append_log(
                log_path,
                [
                    episode,
                    total_step,
                    episode_reward,
                    episode_length,
                    avg_reward_10,
                    latest_losses["critic1_loss"],
                    latest_losses["critic2_loss"],
                    latest_losses["actor_loss"],
                    latest_losses["alpha_loss"],
                    latest_losses["alpha"],
                ],
            )

            progress_bar.set_postfix(
                {
                    "episode": episode,
                    "reward": f"{episode_reward:.1f}",
                    "avg10": f"{avg_reward_10:.1f}",
                    "alpha": f"{latest_losses['alpha']:.3f}",
                }
            )

            obs, info = env.reset(seed=SEED + episode)

            episode_reward = 0.0
            episode_length = 0
            episode += 1

        if total_step % 100_000 == 0:
            checkpoint_dir = os.path.join(SAC_MODEL_DIR, f"checkpoint_{total_step}")
            agent.save(checkpoint_dir)
            print(f"\nSaved SAC checkpoint: {checkpoint_dir}")

    final_model_dir = os.path.join(SAC_MODEL_DIR, "final")
    agent.save(final_model_dir)

    env.close()

    print("\nCustom SAC training complete.")
    print(f"Final model saved to: {final_model_dir}")
    print(f"Training log saved to: {log_path}")


if __name__ == "__main__":
    main()