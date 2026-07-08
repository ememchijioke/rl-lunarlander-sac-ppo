# train_custom_ppo.py

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
    HIDDEN_DIM,
    MAX_EPISODE_STEPS,
    CUSTOM_PPO_TOTAL_STEPS,
    CUSTOM_PPO_ROLLOUT_STEPS,
    CUSTOM_PPO_UPDATE_EPOCHS,
    CUSTOM_PPO_BATCH_SIZE,
    CUSTOM_PPO_LEARNING_RATE,
    CUSTOM_PPO_GAMMA,
    CUSTOM_PPO_GAE_LAMBDA,
    CUSTOM_PPO_CLIP_RANGE,
    CUSTOM_PPO_VALUE_COEF,
    CUSTOM_PPO_ENTROPY_COEF,
    CUSTOM_PPO_MAX_GRAD_NORM,
    CUSTOM_PPO_MODEL_DIR,
    CUSTOM_PPO_LOG_DIR,
)
from env_utils import make_env
from ppo.rollout_buffer import RolloutBuffer
from ppo.ppo_agent import PPOAgent


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
                "policy_loss",
                "value_loss",
                "entropy",
                "total_loss",
            ]
        )


def append_log(log_path, row):
    with open(log_path, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(row)


def main():
    set_seed(SEED)

    os.makedirs(CUSTOM_PPO_MODEL_DIR, exist_ok=True)
    os.makedirs(CUSTOM_PPO_LOG_DIR, exist_ok=True)

    log_path = os.path.join(CUSTOM_PPO_LOG_DIR, "training_log.csv")
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

    agent = PPOAgent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        action_limit=action_limit,
        hidden_dim=HIDDEN_DIM,
        learning_rate=CUSTOM_PPO_LEARNING_RATE,
        gamma=CUSTOM_PPO_GAMMA,
        gae_lambda=CUSTOM_PPO_GAE_LAMBDA,
        clip_range=CUSTOM_PPO_CLIP_RANGE,
        value_coef=CUSTOM_PPO_VALUE_COEF,
        entropy_coef=CUSTOM_PPO_ENTROPY_COEF,
        max_grad_norm=CUSTOM_PPO_MAX_GRAD_NORM,
        device=device,
    )

    rollout_buffer = RolloutBuffer(
        obs_dim=obs_dim,
        action_dim=action_dim,
        size=CUSTOM_PPO_ROLLOUT_STEPS,
        device=device,
    )

    obs, info = env.reset(seed=SEED)

    episode = 1
    episode_reward = 0.0
    episode_length = 0
    recent_rewards = deque(maxlen=10)

    latest_losses = {
        "policy_loss": 0.0,
        "value_loss": 0.0,
        "entropy": 0.0,
        "total_loss": 0.0,
    }

    progress_bar = tqdm(
        range(1, CUSTOM_PPO_TOTAL_STEPS + 1),
        desc="Training Custom Deep PPO",
    )

    for total_step in progress_bar:
        action, logprob, value = agent.select_action(obs)

        next_obs, reward, terminated, truncated, info = env.step(action)

        episode_reward += reward
        episode_length += 1

        timeout = episode_length >= MAX_EPISODE_STEPS
        done = terminated or truncated or timeout

        rollout_buffer.store(
            obs=obs,
            action=action,
            logprob=logprob,
            reward=reward,
            done=float(terminated or truncated),
            value=value,
        )

        obs = next_obs

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
                    latest_losses["policy_loss"],
                    latest_losses["value_loss"],
                    latest_losses["entropy"],
                    latest_losses["total_loss"],
                ],
            )

            progress_bar.set_postfix(
                {
                    "episode": episode,
                    "reward": f"{episode_reward:.1f}",
                    "avg10": f"{avg_reward_10:.1f}",
                }
            )

            obs, info = env.reset(seed=SEED + episode)
            episode_reward = 0.0
            episode_length = 0
            episode += 1

        if rollout_buffer.ptr == CUSTOM_PPO_ROLLOUT_STEPS:
            last_value = 0.0 if done else agent.get_value(obs)

            rollout_buffer.compute_returns_and_advantages(
                last_value=last_value,
                gamma=CUSTOM_PPO_GAMMA,
                gae_lambda=CUSTOM_PPO_GAE_LAMBDA,
            )

            latest_losses = agent.update(
                rollout_buffer=rollout_buffer,
                batch_size=CUSTOM_PPO_BATCH_SIZE,
                update_epochs=CUSTOM_PPO_UPDATE_EPOCHS,
            )

            rollout_buffer.clear()

        if total_step % 100_000 == 0:
            checkpoint_dir = os.path.join(
                CUSTOM_PPO_MODEL_DIR,
                f"checkpoint_{total_step}",
            )
            agent.save(checkpoint_dir)
            print(f"\nSaved Custom PPO checkpoint: {checkpoint_dir}")

    final_model_dir = os.path.join(CUSTOM_PPO_MODEL_DIR, "final")
    agent.save(final_model_dir)

    env.close()

    print("\nCustom Deep PPO training complete.")
    print(f"Final model saved to: {final_model_dir}")
    print(f"Training log saved to: {log_path}")


if __name__ == "__main__":
    main()