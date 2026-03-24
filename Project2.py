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
np.random.seed(41) # set np random seed for reproducibility
gamma = 0.975
prob = 0.7
threshold = 0.001 # try 0.001, 0.01, 0.1

# Task 1: Design the Environment
def step(state, action_idx):
    
    # Stochastic action selection
    if(np.random.uniform(0,1) < prob):
        # intended action is taken with 0.7 probability
        action = ACTIONS[action_idx]
    else:
        # Randomly select an action (simulate stochasticity)
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
def evaluate_policy(policy):
    V = -1 * np.ones((GRID_SIZE, GRID_SIZE))
    V[GOAL_STATE] = 0  # Goal state has a value of 0
    Running = True
    norm = []
    k = 0
    while Running:  # Run for a fixed number of iterations
        V_new = np.zeros((GRID_SIZE, GRID_SIZE))
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                if (i, j) == GOAL_STATE:
                    V_new[i, j] = 0
                elif (i, j) in FENCES:
                    V_new[i, j] = -1
                else:
                    for action_index in range(4): # Iterate through all actions to calculate expected value
                        action = ACTIONS[action_index]
                        
                        if action_index == policy[i, j]: # If action is the intended action, use 0.7 probability
                            prob = 0.7
                        else: # Otherwise, the other 3 actions have equal probability of 0.1
                            prob = 0.1
                        new_state = (i + action[0], j + action[1])
                        
                        # Check boundaries of the grid
                        if (0 <= new_state[0] < GRID_SIZE) and (0 <= new_state[1] < GRID_SIZE):
                        # Check for fences, all states have -1 penalty except goal state without penalty
                            if new_state in FENCES:
                                new_state, reward = (i, j), -1  # Hit a fence, stay in place, normal -1 penalty
                            elif new_state == GOAL_STATE:
                                new_state, reward = new_state, 0  # Reached goal, no penalty
                            else:
                                new_state, reward = new_state, -1  # Valid move, normal -1 penalty
                        else:
                            new_state, reward = (i, j), -1  # Out of bounds, stay in place, normal -1 penalty
                        # Update the value function using the Bellman expectation equation
                        V_new[i, j] += prob * (reward + gamma * V[new_state])
        V_old = np.copy(V)
        V = V_new
        norm.append(np.linalg.norm(V_new - V_old))
        if(norm[-1] < threshold):
            Running = False
        k += 1
    return V, norm

random_policy = np.random.randint(0, 4, size=(GRID_SIZE, GRID_SIZE))  # Generate random policy
U_random, norm = evaluate_policy(random_policy)
# Heatmap of Value Function
plot_heatmap(U_random, "Task 2: Value Function of a Random Policy")
# Norm Plot for Convergence of Policy Evaluation
plt.plot(norm, label='Random Policy')
plt.xlabel('Policy Evaluation Index k')
plt.ylabel('Norm of Value Function Change')
plt.title(f'Convergence of Policy Evaluation, Threshold = {threshold}')
plt.legend()
plt.show()

# Task 3: Localized Policy Optimization
