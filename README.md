#### Academic Coursework
# RL LunarLander: Custom SAC vs PPO

# RL LunarLander SAC PPO

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
Record videos

# Record PPO
python record_video.py --algo ppo --episodes 5 --video-folder videos/ppo_1m

# Record custom SAC
python record_video.py --algo sac --episodes 5 --video-folder videos/sac_1m

Videos are written under videos/.

Plot training curves

python plot_results.py
python compare_results.py

Generated artefacts are written under:
* plots/ppo_training_curve.png
* plots/sac_training_curve.png
* plots/sac_vs_ppo_comparison.png
* plots/evaluation_summary_table.png

Results
Both agents were trained for 1,000,000 timesteps.
### PPO Training Curve

![PPO Training Curve](plots/ppo_training_curve.png)

### Custom SAC Training Curve

![Custom SAC Training Curve](plots/sac_training_curve.png)

### PPO vs Custom SAC Comparison

![PPO vs Custom SAC Comparison](plots/sac_vs_ppo_comparison.png)

### Evaluation Summary

![Evaluation Summary](plots/evaluation_summary_table.png)


Notes
- Custom SAC implementation is inside sac/.
- PPO baseline uses Stable-Baselines3.
- Video recording uses rgb_array rendering because live rendering caused instability on the local Linux system.
- The comparison is based on training reward curves, final evaluation reward, and recorded landing behavior.


License

MIT License