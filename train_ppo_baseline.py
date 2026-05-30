import os

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from config import (
    ENV_NAME,
    SEED,
    PPO_TOTAL_STEPS,
    PPO_MODEL_DIR,
    PPO_LOG_DIR,
    PPO_LEARNING_RATE,
    PPO_N_STEPS,
    PPO_BATCH_SIZE,
    PPO_N_EPOCHS,
    PPO_GAMMA,
    PPO_GAE_LAMBDA,
    PPO_CLIP_RANGE,
)
from env_utils import make_env


def create_training_env():
    env = make_env(ENV_NAME, render_mode=None, seed=SEED)
    env = Monitor(env, filename=os.path.join(PPO_LOG_DIR, "monitor.csv"))
    return env


def main():
    os.makedirs(PPO_MODEL_DIR, exist_ok=True)
    os.makedirs(PPO_LOG_DIR, exist_ok=True)

    env = DummyVecEnv([create_training_env])

    checkpoint_callback = CheckpointCallback(
        save_freq=50_000,
        save_path=PPO_MODEL_DIR,
        name_prefix="ppo_lunarlander_checkpoint",
    )

    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=PPO_LEARNING_RATE,
        n_steps=PPO_N_STEPS,
        batch_size=PPO_BATCH_SIZE,
        n_epochs=PPO_N_EPOCHS,
        gamma=PPO_GAMMA,
        gae_lambda=PPO_GAE_LAMBDA,
        clip_range=PPO_CLIP_RANGE,
        verbose=1,
        seed=SEED,
        tensorboard_log=os.path.join(PPO_LOG_DIR, "tensorboard"),
        policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
    )

    model.learn(
        total_timesteps=PPO_TOTAL_STEPS,
        callback=checkpoint_callback,
        progress_bar=True,
    )

    final_model_path = os.path.join(PPO_MODEL_DIR, "ppo_lunarlander_final")
    model.save(final_model_path)

    env.close()

    print(f"PPO training complete. Model saved to: {final_model_path}.zip")


if __name__ == "__main__":
    main()