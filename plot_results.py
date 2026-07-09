# plot_results.py

import os

import matplotlib.pyplot as plt
import pandas as pd

from config import PLOTS_DIR


CUSTOM_PPO_LOG = "logs/custom_ppo/training_log.csv"
SAC_LOG = "logs/custom_sac/training_log.csv"


def plot_custom_ppo():
    df = pd.read_csv(CUSTOM_PPO_LOG)

    df["avg_reward_100"] = df["episode_reward"].rolling(window=100, min_periods=1).mean()

    plt.figure()
    plt.plot(df["episode"], df["episode_reward"], label="Episode reward", alpha=0.35)
    plt.plot(df["episode"], df["avg_reward_10"], label="10-episode average")
    plt.plot(df["episode"], df["avg_reward_100"], label="100-episode average")
    plt.xlabel("Episode")
    plt.ylabel("Episode Reward")
    plt.title("Custom Deep PPO Training Reward Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    path = os.path.join(PLOTS_DIR, "custom_ppo_training_curve.png")
    plt.savefig(path, dpi=300)
    print(f"Saved: {path}")


def plot_sac():
    df = pd.read_csv(SAC_LOG)

    plt.figure()
    plt.plot(df["episode"], df["episode_reward"], label="Episode reward")
    plt.plot(df["episode"], df["avg_reward_10"], label="10-episode average")
    plt.xlabel("Episode")
    plt.ylabel("Episode Reward")
    plt.title("Custom SAC Training Reward Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    path = os.path.join(PLOTS_DIR, "sac_training_curve.png")
    plt.savefig(path, dpi=300)
    print(f"Saved: {path}")


def plot_comparison():
    ppo_df = pd.read_csv(CUSTOM_PPO_LOG)
    sac_df = pd.read_csv(SAC_LOG)

    ppo_rewards = ppo_df["episode_reward"].reset_index(drop=True)
    sac_rewards = sac_df["episode_reward"].reset_index(drop=True)

    ppo_smooth = ppo_rewards.rolling(window=10, min_periods=1).mean()
    sac_smooth = sac_rewards.rolling(window=10, min_periods=1).mean()

    plt.figure()
    plt.plot(ppo_smooth, label="Custom Deep PPO 10-episode average")
    plt.plot(sac_smooth, label="Custom SAC 10-episode average")
    plt.xlabel("Episode")
    plt.ylabel("Episode Reward")
    plt.title("Custom Deep PPO vs Custom SAC Training Comparison")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    path = os.path.join(PLOTS_DIR, "custom_ppo_vs_sac_comparison.png")
    plt.savefig(path, dpi=300)
    print(f"Saved: {path}")


def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)

    plot_custom_ppo()
    plot_sac()
    plot_comparison()

    print("\nAll plots generated successfully.")


if __name__ == "__main__":
    main()