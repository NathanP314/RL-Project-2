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
plot_heatmap(env_grid, "Task 1: Initial Maze Layout (X = Fence, S = Start, G = Goal)", annot=False)

# Task 2: Policy Evaluation
def evaluate_policy(policy, gamma_value):
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
                        V_new[i, j] += prob * (reward + gamma_value * V[new_state])
        V_old = np.copy(V)
        V = V_new
        norm.append(np.linalg.norm(V_new - V_old))
        if(norm[-1] < threshold):
            Running = False
        k += 1
    return V, norm

random_policy = np.random.randint(0, 4, size=(GRID_SIZE, GRID_SIZE))  # Generate random policy
U_random, norm = evaluate_policy(random_policy, gamma)
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
# Helper to compute Manhattan distance
def manhattan_dist(s1, s2):
    return abs(s1[0] - s2[0]) + abs(s1[1] - s2[1])

# Returns the action index that moves state closer to goal
def optimal_action(state, goal):
    best_action = 0
    min_dist = float('inf')
    for idx, (di, dj) in enumerate(ACTIONS):
        ni, nj = state[0] + di, state[1] + dj
        if 0 <= ni < GRID_SIZE and 0 <= nj < GRID_SIZE and (ni, nj) not in FENCES:
            dist = manhattan_dist((ni, nj), goal)
            if dist < min_dist:
                min_dist = dist
                best_action = idx
    return best_action

# Create the new policy
policy_task3 = np.random.randint(0, 4, size=(GRID_SIZE, GRID_SIZE))
for i in range(GRID_SIZE):
    for j in range(GRID_SIZE):
        if (i, j) == GOAL_STATE or (i, j) in FENCES:
            continue
        if manhattan_dist((i, j), GOAL_STATE) <= 2:
            policy_task3[i, j] = optimal_action((i, j), GOAL_STATE)

# Evaluate and visualize
U_task3, norm3 = evaluate_policy(policy_task3, gamma)
plot_heatmap(U_task3, "Task 3: Value Function with Optimal Region near Goal")

# Task 4: Policy Improvement
def policy_improvement(gamma, threshold, max_iterations):
    # Initialize random policy and value function
    policy = np.random.randint(0, 4, size=(GRID_SIZE, GRID_SIZE))
    V = np.zeros((GRID_SIZE, GRID_SIZE))
    V[GOAL_STATE] = 0

    # Initialize lists for visualization
    norms, V_snapshots, policy_snapshots, snapshot_iters = [], [], [], []

    V_history = []
    policy_history = []
    for it in range(max_iterations):
        # Policy Evaluation
        V_new, norm = evaluate_policy(policy, gamma)
        norms.append(np.linalg.norm(V_new - V))
        V = V_new
        V_history.append(np.copy(V))
        policy_history.append(np.copy(policy))

        # Policy Improvement
        policy_stable = True
        new_policy = np.copy(policy)
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                # Skips updating policy for goal and fence states
                if (i, j) == GOAL_STATE or (i, j) in FENCES:
                    continue

                action_values = []
                for action_idx in range(4):
                    total = 0

                    # Loop through possible outcomes
                    for a_idx in range(4):
                        if a_idx == action_idx:
                            prob_a = 0.7
                        else:
                            prob_a = 0.1

                        # Compute next state
                        a = ACTIONS[a_idx]
                        i_new, j_new = i + a[0], j + a[1]

                        # Handle boundaries and fences
                        if not (0 <= i_new < GRID_SIZE and 0 <= j_new < GRID_SIZE) or (i_new, j_new) in FENCES:
                            new_state, reward = (i, j), -1
                        elif (i_new, j_new) == GOAL_STATE:
                            new_state, reward = (i_new, j_new), 0
                        else:
                            new_state, reward = (i_new, j_new), -1

                        total += prob_a * (reward + gamma * V[new_state])
                    action_values.append(total)
                
                best_action = np.argmax(action_values)

                if best_action != policy[i, j]:
                    policy_stable = False

                new_policy[i, j] = best_action

        policy = new_policy
        if norms[-1] < threshold and policy_stable:
            break
    # Always take snapshots at 0, middle, and final iteration
    total_iterations = len(V_history)
    snapshot_indices = [0, total_iterations // 2, total_iterations - 1]
    V_snapshots = [V_history[idx] for idx in snapshot_indices]
    policy_snapshots = [policy_history[idx] for idx in snapshot_indices]
    snapshot_iters = snapshot_indices
    return V, policy, norms, V_snapshots, policy_snapshots, snapshot_iters

def plot_policy_arrows(policy, ax):
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            if (i, j) == GOAL_STATE or (i, j) in FENCES:
                continue

            action = policy[i, j]

            # Arrow mapping
            dx, dy = 0, 0
            if action == 0: dx = -0.3   # Up
            if action == 1: dy = 0.3    # Right
            if action == 2: dx = 0.3    # Down
            if action == 3: dy = -0.3   # Left

            ax.arrow(j + 0.5, i + 0.5, dy, dx, head_width = 0.2, 
                     head_length = 0.2, fc = 'black', ec = 'black')

# Run Policy Iteration for gamma=0.975
V_opt, policy_opt, norms, V_snaps, policy_snaps, snap_iters = policy_improvement(0.975, threshold, 30)

# Visualize value function and policy at 3 iterations
for idx, (V_i, pi_i, it) in enumerate(zip(V_snaps, policy_snaps, snap_iters)):
    plt.figure(figsize=(8,8))
    ax = sns.heatmap(V_i, annot=True, fmt=".1f", cmap="viridis", cbar=True, linewidths=0.5, linecolor='black', square=True)
    for f in FENCES:
        ax.text(f[1]+0.5, f[0]+0.5, 'X', color='red', ha='center', va='center', fontsize=14, weight='bold')
    ax.text(GOAL_STATE[1]+0.5, GOAL_STATE[0]+0.5, 'G', color='white', ha='center', va='center', fontsize=14, weight='bold')
    ax.text(START_STATE[1]+0.5, START_STATE[0]+0.5, 'S', color='white', ha='center', va='center', fontsize=14, weight='bold')
    plot_policy_arrows(pi_i, ax)
    plt.title(f"Task 4: Value Function & Policy at Iteration {it} (gamma=0.975)")
    plt.show()

# Plot norm of value function difference
plt.figure(figsize=(7,4))
plt.plot(norms, marker='o')
plt.xlabel('Policy Iteration Index i')
plt.ylabel(r'$||V^{\pi_{i+1}} - V^{\pi_i}||_2$')
plt.title('Convergence of Policy Iteration (gamma = 0.975)')
plt.grid(True)
plt.show()

# Repeat with gamma = 0.75
V_opt2, policy_opt2, norms2, V_snaps2, policy_snaps2, snap_iters2 = policy_improvement(0.75, threshold, 30)
for idx, (V_i, pi_i, it) in enumerate(zip(V_snaps2, policy_snaps2, snap_iters2)):
    plt.figure(figsize=(8,8))
    ax = sns.heatmap(V_i, annot=True, fmt=".1f", cmap="viridis", cbar=True, linewidths=0.5, linecolor='black', square=True)
    for f in FENCES:
        ax.text(f[1]+0.5, f[0]+0.5, 'X', color='red', ha='center', va='center', fontsize=14, weight='bold')
    ax.text(GOAL_STATE[1]+0.5, GOAL_STATE[0]+0.5, 'G', color='white', ha='center', va='center', fontsize=14, weight='bold')
    ax.text(START_STATE[1]+0.5, START_STATE[0]+0.5, 'S', color='white', ha='center', va='center', fontsize=14, weight='bold')
    plot_policy_arrows(pi_i, ax)
    plt.title(f"Task 4: Value Function & Policy at Iteration {it} (gamma=0.75)")
    plt.show()

# Plot norm for gamma=0.75
plt.figure(figsize=(7,4))
plt.plot(norms2, marker='o', color='orange')
plt.xlabel('Policy Iteration Index i')
plt.ylabel(r'$||V^{\pi_{i+1}} - V^{\pi_i}||_2$')
plt.title('Convergence of Policy Iteration (gamma = 0.75)')
plt.grid(True)
plt.show()