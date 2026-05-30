import gymnasium as gym
from config import ENV_NAME

env = gym.make(ENV_NAME, render_mode="human")

obs, info = env.reset()

done = False
episode_reward = 0

while not done:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    episode_reward += reward
    done = terminated or truncated

env.close()

print(f"Random policy episode reward: {episode_reward:.2f}")