# evaluate.py

import argparse
import os
import time

import numpy as np
from stable_baselines3 import PPO

from config import ENV_NAME, SEED, PPO_MODEL_DIR, EVAL_EPISODES
from env_utils import make_env


def evaluate_ppo(model_path: str, render: bool = False, episodes: int = EVAL_EPISODES):
    render_mode = "human" if render else None
    env = make_env(ENV_NAME, render_mode=render_mode, seed=SEED)

    model = PPO.load(model_path)

    episode_rewards = []
    episode_lengths = []

    for ep in range(episodes):
        obs, info = env.reset(seed=SEED + ep)
        done = False
        total_reward = 0.0
        steps = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            if render:
                time.sleep(1/60)
            total_reward += reward
            steps += 1
            done = terminated or truncated

        episode_rewards.append(total_reward)
        episode_lengths.append(steps)

        print(f"Episode {ep + 1}: reward={total_reward:.2f}, steps={steps}")

    env.close()

    print("\nEvaluation Summary")
    print("------------------")
    print(f"Model: {model_path}")
    print(f"Episodes: {episodes}")
    print(f"Mean reward: {np.mean(episode_rewards):.2f}")
    print(f"Std reward: {np.std(episode_rewards):.2f}")
    print(f"Best reward: {np.max(episode_rewards):.2f}")
    print(f"Worst reward: {np.min(episode_rewards):.2f}")
    print(f"Mean episode length: {np.mean(episode_lengths):.2f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default=os.path.join(PPO_MODEL_DIR, "ppo_lunarlander_final.zip"),
        help="Path to trained PPO model.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=EVAL_EPISODES,
        help="Number of evaluation episodes.",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render the environment during evaluation.",
    )

    args = parser.parse_args()

    evaluate_ppo(
        model_path=args.model,
        render=args.render,
        episodes=args.episodes,
    )


if __name__ == "__main__":
    main()