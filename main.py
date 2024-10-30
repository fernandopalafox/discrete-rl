import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map

env = gym.make(
    "FrozenLake-v1", map_name="4x4", is_slippery=True, desc=generate_random_map(size=8)
)

print(env.observation_space)
print(env.action_space)