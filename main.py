import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
import numpy as np
import matplotlib.pyplot as plt


# Params
rng_seed = 42
env_size = 4
num_episodes = 10000
alpha = 0.02
epsilon_0 = 1.0
epsilon_decay = 0.999
epsilon_min = 0.05
discount_factor = 0.9

# Generate env
env = gym.make(
    "FrozenLake-v1",
    map_name="4x4",
    is_slippery=False,
    desc=generate_random_map(size=env_size, seed=rng_seed),
)


# Reward should go up lol
Q = np.zeros([env.observation_space.n, env.action_space.n])

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


def plot_everything(Q, env):
    """
    Plot the environment map, the optimal policy with grid cells colored according to the Q-value
    of the optimal action at each state, and the full Q table.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    env_size = env.unwrapped.desc.shape[0]
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))

    # First subplot: Map visualization
    # Get the map description
    desc = env.unwrapped.desc
    desc_chars = np.array([[c.decode('utf-8') for c in line] for line in desc])

    # Map the characters to integers
    char_to_int = {'S': 0, 'F': 1, 'H': 2, 'G': 3}
    map_array = np.vectorize(char_to_int.get)(desc_chars)

    # Create a color map
    cmap = ListedColormap(['green', 'lightblue', 'black', 'gold'])

    # Plot the map
    ax[0].imshow(map_array, cmap=cmap, origin='upper')
    ax[0].set_xticks(np.arange(env_size))
    ax[0].set_yticks(np.arange(env_size))
    ax[0].set_xticklabels([])
    ax[0].set_yticklabels([])
    ax[0].set_xlim(-0.5, env_size - 0.5)
    ax[0].set_ylim(-0.5, env_size - 0.5)
    ax[0].invert_yaxis()
    ax[0].set_title("Environment Map")

    # Add text labels to indicate the type of each cell
    for i in range(env_size):
        for j in range(env_size):
            cell_text = desc_chars[i, j]
            if cell_text == 'F':
                cell_text = ''
            color = 'white' if map_array[i, j] in [2] else 'black'  # Use white text on black holes
            ax[0].text(j, i, cell_text, ha='center', va='center', color=color, fontsize=12)

    # Second subplot: Optimal policy with Q-values
    # Compute the optimal policy and the corresponding Q-values
    optimal_policy = np.argmax(Q, axis=1)
    optimal_Q_values = np.max(Q, axis=1)

    # Reshape for grid plotting
    optimal_policy_grid = optimal_policy.reshape(env_size, env_size)
    optimal_Q_values_grid = optimal_Q_values.reshape(env_size, env_size)

    # Prepare grid for arrows
    x = np.arange(env_size)
    y = np.arange(env_size)
    X, Y = np.meshgrid(x, y)

    U = np.zeros_like(optimal_policy_grid, dtype=float)
    V = np.zeros_like(optimal_policy_grid, dtype=float)

    for i in range(env_size):
        for j in range(env_size):
            action = optimal_policy_grid[i, j]
            if i == env_size-1 and j == env_size-1:
                continue # Skip goal
            elif map_array[i, j] == 2:
                continue
            elif action == 0:  # Move left
                U[i, j] = -1
                V[i, j] = 0
            elif action == 1:  # Move down
                U[i, j] = 0
                V[i, j] = 1
            elif action == 2:  # Move right
                U[i, j] = 1
                V[i, j] = 0
            elif action == 3:  # Move up
                U[i, j] = 0
                V[i, j] = -1

    # Plot the optimal Q-values as background
    c = ax[1].imshow(optimal_Q_values_grid, origin='upper', cmap='viridis')
    # fig.colorbar(c, ax=ax[1], fraction=0.046, pad=0.04)
    ax[1].set_xticks(np.arange(env_size))
    ax[1].set_yticks(np.arange(env_size))
    ax[1].set_xticklabels([])
    ax[1].set_yticklabels([])
    ax[1].set_xlim(-0.5, env_size - 0.5)
    ax[1].set_ylim(-0.5, env_size - 0.5)

    # Overlay the arrows
    ax[1].quiver(X, Y, U, V, angles='xy', scale_units='xy', scale=1, color='white')
    ax[1].invert_yaxis()  # Flip y-axis to match grid indexing
    ax[1].set_title("Value Function and Optimal Policy")

    # Mark start and end points
    ax[1].scatter(0, 0, s=100, c='green', marker='o', label='Start')
    ax[1].scatter(env_size - 1, env_size - 1, s=100, c='gold', marker='X', label='Goal')
    ax[1].legend(loc='upper right')

    # Third subplot: Q table
    cax = ax[2].imshow(Q, aspect='auto', cmap="viridis")
    ax[2].set_title("Q Table")
    ax[2].set_xlabel("Actions")
    ax[2].set_ylabel("State")
    fig.colorbar(cax, ax=ax[2], fraction=0.046, pad=0.04)
    ax[2].set_xticks(np.arange(env.action_space.n))
    ax[2].set_xticklabels(["L", "D", "R", "U"])  # Left, Down, Right, Up

    plt.tight_layout()
    plt.savefig("figures/Q.png")

epsilon = epsilon_0
for i in range(num_episodes):
    episode, total_reward = generate_episode(env, Q, epsilon)
    Q = update_Q(Q, episode, alpha, discount_factor)
    epsilon = max(epsilon * epsilon_decay, epsilon_min)
    print(f"Ep {i+1}/{num_episodes}, eps: {epsilon}, TR: {total_reward}")

plot_everything(Q, env)