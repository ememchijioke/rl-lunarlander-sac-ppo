# record_video.py

import argparse
import os

import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import torch
from stable_baselines3 import PPO

from config import (
    ENV_NAME,
    SEED,
    PPO_MODEL_DIR,
    SAC_MODEL_DIR,
    VIDEOS_DIR,
    HIDDEN_DIM,
    ACTOR_LR,
    CRITIC_LR,
    ALPHA_LR,
    GAMMA,
    TAU,
)
from sac.sac_agent import SACAgent


def make_video_env(video_folder, name_prefix):
    os.makedirs(video_folder, exist_ok=True)

    env = gym.make(ENV_NAME, render_mode="rgb_array")

    env = RecordVideo(
        env,
        video_folder=video_folder,
        name_prefix=name_prefix,
        episode_trigger=lambda episode_id: True,
    )

    return env


def load_sac_agent(env, model_dir):
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
    return agent


def record_ppo(model_path, episodes, video_folder):
    env = make_video_env(video_folder, "ppo_lunarlander")
    model = PPO.load(model_path)

    run_episodes(env, episodes, lambda obs: model.predict(obs, deterministic=True)[0])

    env.close()
    print(f"\nPPO video recording complete. Saved in: {video_folder}")


def record_sac(model_dir, episodes, video_folder):
    env = make_video_env(video_folder, "sac_lunarlander")
    agent = load_sac_agent(env, model_dir)

    run_episodes(env, episodes, lambda obs: agent.select_action(obs, deterministic=True))

    env.close()
    print(f"\nSAC video recording complete. Saved in: {video_folder}")


def run_episodes(env, episodes, action_fn):
    for ep in range(episodes):
        obs, info = env.reset(seed=SEED + ep)

        done = False
        total_reward = 0.0
        steps = 0

        while not done:
            action = action_fn(obs)
            obs, reward, terminated, truncated, info = env.step(action)

            total_reward += reward
            steps += 1
            done = terminated or truncated

        print(f"Episode {ep + 1}: reward={total_reward:.2f}, steps={steps}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--algo",
        type=str,
        choices=["ppo", "sac"],
        required=True,
        help="Algorithm to record.",
    )

    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Optional model path. PPO expects .zip, SAC expects model directory.",
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
        default=None,
        help="Folder where videos will be saved.",
    )

    args = parser.parse_args()

    if args.algo == "ppo":
        model_path = args.model or os.path.join(PPO_MODEL_DIR, "ppo_lunarlander_final.zip")
        video_folder = args.video_folder or os.path.join(VIDEOS_DIR, "ppo")
        record_ppo(model_path, args.episodes, video_folder)

    elif args.algo == "sac":
        model_dir = args.model or os.path.join(SAC_MODEL_DIR, "final")
        video_folder = args.video_folder or os.path.join(VIDEOS_DIR, "sac")
        record_sac(model_dir, args.episodes, video_folder)


if __name__ == "__main__":
    main()