import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Environment Constants
GRID_SIZE = 10
GOAL_STATE = (0, 9)
FENCES = [(1, 6), (3, 7), (7,9)]
START_STATE = (9, 0)

# Actions: 0: Up, 1: Right, 2: Down, 3: Left
ACTIONS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
ACTION_SYMBOLS = ['↑', '→', '↓', '←']

# Other Constants
gamma = 0.975
prob = 0.7

# Task 1: Design the Environment
def step(state, action_idx):
    
    # Stochastic action selection
    if(np.random.uniform(0,1) > prob):
        # intended action is taken with 0.7 probability
        action = ACTIONS[action_idx]
    else:
        # Randomly select an adjacent action (simulate stochasticity)
        while action != ACTIONS[action_idx]:
            action = ACTIONS[np.random.randint(0, 4)]
            
    new_state = (state[0] + action[0], state[1] + action[1])
    
    # Check boundaries of the grid
    if (0 <= new_state[0] < GRID_SIZE) and (0 <= new_state[1] < GRID_SIZE):
      # Check for fences, all states have -1 penalty except goal state without penalty
        if new_state in FENCES:
            return state, -1, False  # Hit a fence, stay in place, normal -1 penalty
        elif new_state == GOAL_STATE:
            return new_state, 0, True  # Reached goal, no penalty
        else:
            return new_state, -1, False  # Valid move, normal -1 penalty
    else:
        return state, -1, False  # Out of bounds, stay in place, normal -1 penalty

def plot_heatmap(data, title, annot=True, fmt=".1f"):
    """Helper to plot utilities with a visible grid structure."""
    plt.figure(figsize=(8, 8)) # Increased size for clarity
    mask = np.zeros_like(data, dtype=bool)
    for f in FENCES:
        mask[f] = True

    # linewidths adds the border, linecolor sets the color of the grid lines
    ax = sns.heatmap(data, annot=annot, fmt=fmt, cmap="viridis", mask=mask, cbar=True, linewidths=0.5, linecolor='black',square=True)

    # Mark Fences and Goal
    for f in FENCES:
        ax.text(f[1]+0.5, f[0]+0.5, 'X', color='red', ha='center', va='center', fontsize=14, weight='bold')
    ax.text(GOAL_STATE[1]+0.5, GOAL_STATE[0]+0.5, 'G', color='white', ha='center', va='center', fontsize=14, weight='bold')
    ax.text(START_STATE[1]+0.5, START_STATE[0]+0.5, 'S', color='white', ha='center', va='center', fontsize=14, weight='bold')
    
    plt.title(title)
    plt.show()

# Visualize Task 1
env_grid = np.zeros((GRID_SIZE, GRID_SIZE))
plot_heatmap(env_grid, "Task 1: Initial Maze Layout (X = Fence, G = Goal)", annot=False)

# Task 2: Policy Evaluation
