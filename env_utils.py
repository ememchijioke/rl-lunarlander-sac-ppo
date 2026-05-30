# env_utils.py

import gymnasium as gym


def make_env(env_name: str, render_mode=None, seed: int = 42):
    """
    Create the LunarLanderContinuous environment for v3.

    This helper will keep the environment creation consistent across training,
    evaluation, plotting, and video recording.
    """

    env = gym.make(env_name, render_mode=render_mode)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)

    return env