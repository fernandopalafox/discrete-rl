import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
import numpy as np
from plot import plot_everything


def generate_episode(env, Q, epsilon):
    """
    Episode generator using epsilon-greedy policy.
    Returns a list of (state, action, reward) tuples
    """
    state, _ = env.reset()
    episode = []
    total_reward = 0
    while True:
        if np.random.uniform() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state, :])
        next_state, reward, done, _, _ = env.step(action)
        episode.append((state, action, reward))
        state = next_state
        total_reward += reward
        if done:
            break
    return episode, total_reward


def update_Q(Q, episode, alpha, discount_factor):
    """
    Update Q table using first-visit constant alpha MC update rule
    """
    visited = set()
    G = 0
    for state, action, reward in episode[::-1]:
        G = discount_factor * G + reward
        if (state, action) not in visited:
            visited.add((state, action))
            Q[state, action] = Q[state, action] + alpha * (G - Q[state, action])

    return Q


# Params
rng_seed = 42
env_size = 8
num_episodes = 1000000
alpha = 0.005
epsilon_0 = 1.0
epsilon_decay = 0.9999
epsilon_min = 0.05
discount_factor = 0.6
is_slippery = True

# Run
env = gym.make(
    "FrozenLake-v1",
    map_name="4x4",
    is_slippery=is_slippery,
    desc=generate_random_map(size=env_size, seed=rng_seed),
)
np.random.seed(rng_seed)
Q = np.zeros([env.observation_space.n, env.action_space.n])
epsilon = epsilon_0
for i in range(num_episodes):
    episode, total_reward = generate_episode(env, Q, epsilon)
    Q = update_Q(Q, episode, alpha, discount_factor)
    epsilon = max(epsilon * epsilon_decay, epsilon_min)
    print(f"Ep {i+1}/{num_episodes}, eps: {epsilon}, TR: {total_reward}")

plot_everything(Q, env)
