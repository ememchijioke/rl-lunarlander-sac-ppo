# evaluate.py

import argparse
import os

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO

from config import (
    ENV_NAME,
    SEED,
    EVAL_EPISODES,
    PPO_MODEL_DIR,
    SAC_MODEL_DIR,
    HIDDEN_DIM,
    ACTOR_LR,
    CRITIC_LR,
    ALPHA_LR,
    GAMMA,
    TAU,
)
from sac.sac_agent import SACAgent


def make_eval_env(seed):
    env = gym.make(ENV_NAME)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return env


def evaluate_ppo(model_path, episodes):
    env = make_eval_env(SEED)
    model = PPO.load(model_path)

    rewards = []
    lengths = []

    for ep in range(episodes):
        obs, info = env.reset(seed=SEED + ep)
        done = False
        total_reward = 0.0
        steps = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            total_reward += reward
            steps += 1
            done = terminated or truncated

        rewards.append(total_reward)
        lengths.append(steps)
        print(f"Episode {ep + 1}: reward={total_reward:.2f}, steps={steps}")

    env.close()
    print_summary("PPO", model_path, rewards, lengths)


def evaluate_sac(model_dir, episodes):
    env = make_eval_env(SEED)

    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_limit = float(env.action_space.high[0])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    agent.load(model_dir)

    rewards = []
    lengths = []

    for ep in range(episodes):
        obs, info = env.reset(seed=SEED + ep)
        done = False
        total_reward = 0.0
        steps = 0

        while not done:
            action = agent.select_action(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            total_reward += reward
            steps += 1
            done = terminated or truncated

        rewards.append(total_reward)
        lengths.append(steps)
        print(f"Episode {ep + 1}: reward={total_reward:.2f}, steps={steps}")

    env.close()
    print_summary("Custom SAC", model_dir, rewards, lengths)


def print_summary(name, model_path, rewards, lengths):
    print("\nEvaluation Summary")
    print("------------------")
    print(f"Algorithm: {name}")
    print(f"Model: {model_path}")
    print(f"Episodes: {len(rewards)}")
    print(f"Mean reward: {np.mean(rewards):.2f}")
    print(f"Std reward: {np.std(rewards):.2f}")
    print(f"Best reward: {np.max(rewards):.2f}")
    print(f"Worst reward: {np.min(rewards):.2f}")
    print(f"Mean episode length: {np.mean(lengths):.2f}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--algo",
        type=str,
        choices=["ppo", "sac"],
        required=True,
        help="Algorithm to evaluate.",
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=EVAL_EPISODES,
        help="Number of evaluation episodes.",
    )

    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Optional custom model path.",
    )

    args = parser.parse_args()

    if args.algo == "ppo":
        model_path = args.model or os.path.join(PPO_MODEL_DIR, "ppo_lunarlander_final.zip")
        evaluate_ppo(model_path, args.episodes)

    elif args.algo == "sac":
        model_dir = args.model or os.path.join(SAC_MODEL_DIR, "final")
        evaluate_sac(model_dir, args.episodes)


if __name__ == "__main__":
    main()