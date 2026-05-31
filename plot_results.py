# plot_results.py

import os
import pandas as pd
import matplotlib.pyplot as plt

from config import PLOTS_DIR


PPO_LOG = "logs/ppo/monitor.csv"
SAC_LOG = "logs/custom_sac/training_log.csv"


def plot_ppo():
    df = pd.read_csv(PPO_LOG, skiprows=1)

    plt.figure()
    plt.plot(df["r"])
    plt.xlabel("Episode")
    plt.ylabel("Episode Reward")
    plt.title("PPO Training Reward Curve")
    plt.grid(True)
    plt.tight_layout()

    path = os.path.join(PLOTS_DIR, "ppo_training_curve.png")
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
    ppo_df = pd.read_csv(PPO_LOG, skiprows=1)
    sac_df = pd.read_csv(SAC_LOG)

    ppo_rewards = ppo_df["r"].reset_index(drop=True)
    sac_rewards = sac_df["episode_reward"].reset_index(drop=True)

    ppo_smooth = ppo_rewards.rolling(window=10, min_periods=1).mean()
    sac_smooth = sac_rewards.rolling(window=10, min_periods=1).mean()

    plt.figure()
    plt.plot(ppo_smooth, label="PPO 10-episode average")
    plt.plot(sac_smooth, label="Custom SAC 10-episode average")
    plt.xlabel("Episode")
    plt.ylabel("Episode Reward")
    plt.title("PPO vs Custom SAC Training Comparison")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    path = os.path.join(PLOTS_DIR, "sac_vs_ppo_comparison.png")
    plt.savefig(path, dpi=300)
    print(f"Saved: {path}")


def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)

    plot_ppo()
    plot_sac()
    plot_comparison()

    print("\nAll plots generated successfully.")


if __name__ == "__main__":
    main()