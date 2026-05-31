#### Academic Coursework

# RL LunarLander SAC and PPO

This repository is a project that uses deep reinforcement learning for continuous lunar landing control. With a custom Soft Actor-Critic (SAC) agent and a PPO baseline, the lunar lander learns to stabilize its descent and land safely in the LunarLanderContinuous environment.

- **Environment:** LunarLanderContinuous-v3
- **Objective:** learn continuous engine control for safe landing
- **Algorithms:** Custom SAC and PPO baseline
- **Hardware:** trained locally with CUDA support, CPU fallback was available

## Setup

```bash
git clone https://github.com/ememchijioke/rl-lunarlander-sac-ppo.git
cd rl-lunarlander-sac-ppo

python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt

# Train PPO baseline
python train_ppo_baseline.py

# Train custom SAC
python train_custom_sac.py

# Record PPO
python record_video.py --algo ppo --episodes 5 --video-folder videos/ppo_1m

# Record custom SAC
python record_video.py --algo sac --episodes 5 --video-folder videos/sac_1m
# plot training curves
python plot_results.py
python compare_results.py


## Results

Both agents were trained for **1,000,000 timesteps**.

| Algorithm | Mean Reward | Best Reward | Worst Reward |
|---|---:|---:|---:|
| PPO | 195.38 | 266.63 | 25.41 |
| Custom SAC | 233.57 | 282.58 | 16.45 |

## PPO Training Curve

<p align="center">
  <img src="plots/ppo_training_curve.png" width="650">
</p>

## Custom SAC Training Curve

<p align="center">
  <img src="plots/sac_training_curve.png" width="650">
</p>

## PPO vs Custom SAC Comparison

<p align="center">
  <img src="plots/sac_vs_ppo_comparison.png" width="650">
</p>

## Evaluation Summary

<p align="center">
  <img src="plots/evaluation_summary_table.png" width="750">
</p>