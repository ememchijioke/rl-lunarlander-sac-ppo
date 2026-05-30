
import argparse
import os

import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import PPO

from config import ENV_NAME, SEED, PPO_MODEL_DIR, VIDEOS_DIR


def record_ppo_video(model_path: str, episodes: int, video_folder: str):
    os.makedirs(video_folder, exist_ok=True)

    env = gym.make(ENV_NAME, render_mode="rgb_array")

    env = RecordVideo(
        env,
        video_folder=video_folder,
        name_prefix="ppo_lunarlander",
        episode_trigger=lambda episode_id: True,
    )

    model = PPO.load(model_path)

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

        print(f"Episode {ep + 1}: reward={total_reward:.2f}, steps={steps}")

    env.close()

    print(f"\nVideo recording complete.")
    print(f"Saved videos in: {video_folder}")


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
        default=5,
        help="Number of episodes to record.",
    )

    parser.add_argument(
        "--video-folder",
        type=str,
        default=os.path.join(VIDEOS_DIR, "ppo"),
        help="Folder where videos will be saved.",
    )

    args = parser.parse_args()

    record_ppo_video(
        model_path=args.model,
        episodes=args.episodes,
        video_folder=args.video_folder,
    )


if __name__ == "__main__":
    main()