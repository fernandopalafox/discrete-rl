import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
import random

# Params
rng_seed = 42
env_size = 4
num_episodes = 3

# Generate env
env = gym.make(
    "FrozenLake-v1",
    map_name="4x4",
    is_slippery=True,
    desc=generate_random_map(size=env_size, seed=rng_seed),
)

for i_episode in range(num_episodes):
    state, _ = env.reset()
    print('Initial state:', state)

    while True:
        action = env.action_space.sample() 
        state, reward, done, truncated, info = env.step(action)

        # Print stuff
        print('Action:', action)
        print('State:', state)
        print('Reward:', reward)
        print('Done:', done)