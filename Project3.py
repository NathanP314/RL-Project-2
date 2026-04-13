import numpy as np
import matplotlib.pyplot as plt

# Environment Constants
GRID_ROWS = 5
GRID_COLS = 10
GOAL_STATE = (0, GRID_COLS - 1)
START_STATE = (GRID_ROWS - 1, 0)
FENCES = []
CLIFF_STATES = [(GRID_ROWS - 1, 3), (GRID_ROWS - 1, 4), (GRID_ROWS - 1, 5)]

# Actions: 0: Up, 1: Right, 2: Down, 3: Left
ACTIONS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
ACTION_SYMBOLS = ['↑', '→', '↓', '←']

# Other Constants
np.random.seed(41)
gamma = 0.975
prob = 0.7
threshold = 0.001

# Task 1: Step Function (baseline + cliff dynamics)
def step(state, action_idx):
    # Goal is terminal
    if state == GOAL_STATE:
        return state, 0

    # Stochastic action selection (same style as HW2)
    if np.random.uniform(0, 1) < prob:
        action = ACTIONS[action_idx]
    else:
        other_actions = [0, 1, 2, 3]
        other_actions.remove(action_idx)
        action = ACTIONS[np.random.choice(other_actions)]

    new_state = (state[0] + action[0], state[1] + action[1])

    # Check boundaries
    if (0 <= new_state[0] < GRID_ROWS) and (0 <= new_state[1] < GRID_COLS):
        # Check fence
        if new_state in FENCES:
            return state, -1
        # Check cliff
        if new_state in CLIFF_STATES:
            return START_STATE, -100
        # Check goal
        if new_state == GOAL_STATE:
            return new_state, 0
        # Normal move
        return new_state, -1
    else:
        # Out of bounds
        return state, -1

print("Grid:", GRID_ROWS, "x", GRID_COLS)
print("Start:", START_STATE, "Goal:", GOAL_STATE)
print("Cliff:", CLIFF_STATES)

# Task 6 support: compute optimal policy using HW2-style policy iteration
policy_opt = np.random.randint(0, 4, size=(GRID_ROWS, GRID_COLS))
V = np.zeros((GRID_ROWS, GRID_COLS))

for _ in range(50):
    # Policy Evaluation
    running = True
    while running:
        V_new = np.copy(V)
        for i in range(GRID_ROWS):
            for j in range(GRID_COLS):
                s = (i, j)
                if s == GOAL_STATE or s in FENCES:
                    V_new[s] = 0
                    continue

                a = policy_opt[i, j]
                total = 0
                for sampled_action in range(4):
                    p = 0.7 if sampled_action == a else 0.1
                    move = ACTIONS[sampled_action]
                    ns = (i + move[0], j + move[1])

                    if (0 <= ns[0] < GRID_ROWS) and (0 <= ns[1] < GRID_COLS):
                        if ns in FENCES:
                            next_state, reward = s, -1
                        elif ns in CLIFF_STATES:
                            next_state, reward = START_STATE, -100
                        elif ns == GOAL_STATE:
                            next_state, reward = ns, 0
                        else:
                            next_state, reward = ns, -1
                    else:
                        next_state, reward = s, -1

                    total += p * (reward + gamma * V[next_state])

                V_new[s] = total

        if np.linalg.norm(V_new - V) < threshold:
            running = False
        V = V_new

    # Policy Improvement
    policy_stable = True
    for i in range(GRID_ROWS):
        for j in range(GRID_COLS):
            s = (i, j)
            if s == GOAL_STATE or s in FENCES:
                continue

            action_values = []
            for a in range(4):
                total = 0
                for sampled_action in range(4):
                    p = 0.7 if sampled_action == a else 0.1
                    move = ACTIONS[sampled_action]
                    ns = (i + move[0], j + move[1])

                    if (0 <= ns[0] < GRID_ROWS) and (0 <= ns[1] < GRID_COLS):
                        if ns in FENCES:
                            next_state, reward = s, -1
                        elif ns in CLIFF_STATES:
                            next_state, reward = START_STATE, -100
                        elif ns == GOAL_STATE:
                            next_state, reward = ns, 0
                        else:
                            next_state, reward = ns, -1
                    else:
                        next_state, reward = s, -1

                    total += p * (reward + gamma * V[next_state])

                action_values.append(total)

            best_action = np.argmax(action_values)
            if best_action != policy_opt[i, j]:
                policy_stable = False
            policy_opt[i, j] = best_action

    if policy_stable:
        break

# Same 10 initial states for both policies
valid_states = []
for i in range(GRID_ROWS):
    for j in range(GRID_COLS):
        s = (i, j)
        if s != GOAL_STATE and s not in CLIFF_STATES and s not in FENCES:
            valid_states.append(s)

idx = np.random.choice(len(valid_states), size=10, replace=False)
initial_states = [valid_states[k] for k in idx]

random_runs = []
optimal_runs = []

# 10 trajectories with random policy
for s0 in initial_states:
    state = s0
    trajectory = [state]
    rewards = []

    for _ in range(100):
        if state == GOAL_STATE:
            break
        action_idx = np.random.randint(0, 4)
        next_state, reward = step(state, action_idx)
        rewards.append(reward)
        trajectory.append(next_state)
        state = next_state
        if state == GOAL_STATE:
            break

    random_runs.append({"start": s0, "trajectory": trajectory, "rewards": rewards})

# 10 trajectories with optimal policy from task 4 HW 2
for s0 in initial_states:
    state = s0
    trajectory = [state]
    rewards = []

    for _ in range(100):
        if state == GOAL_STATE:
            break
        action_idx = int(policy_opt[state[0], state[1]])
        next_state, reward = step(state, action_idx)
        rewards.append(reward)
        trajectory.append(next_state)
        state = next_state
        if state == GOAL_STATE:
            break

    optimal_runs.append({"start": s0, "trajectory": trajectory, "rewards": rewards})

print("\n" + "=" * 40)
print("10 Trajectories with Random Policy")
print("=" * 40)
for i, item in enumerate(random_runs, start=1):
    total_reward = int(np.sum(item["rewards"]))
    steps = len(item["trajectory"]) - 1
    reached_goal = item["trajectory"][-1] == GOAL_STATE
    print(
        f"Traj {i:02d} | start={item['start']} | steps={steps:02d} | "
        f"return={total_reward:>4d} | reached_goal={reached_goal}"
    )

print("\n" + "=" * 40)
print("10 Trajectories with Optimal Policy")
print("=" * 40)
for i, item in enumerate(optimal_runs, start=1):
    total_reward = int(np.sum(item["rewards"]))
    steps = len(item["trajectory"]) - 1
    reached_goal = item["trajectory"][-1] == GOAL_STATE
    print(
        f"Traj {i:02d} | start={item['start']} | steps={steps:02d} | "
        f"return={total_reward:>4d} | reached_goal={reached_goal}"
    )

# Task 2: Q-Learning
def q_learning(episodes = 200, alpha=0.1, epsilon=0.1):
    Q_table = np.zeros((GRID_ROWS, GRID_COLS, 4))
    e_rewards = [] # track rewards per episode for plotting
    for e in range(episodes): # iterate over number of episodes
        state = START_STATE
        total_rewards = 0
        for step_idx in range(500): # iterate over max 500 steps per episode
            if state == GOAL_STATE:
                break

            # Epsilon-greedy action selection
            if np.random.uniform(0, 1) < epsilon: # with probability episilon, select random action
                action = np.random.randint(0, 4)
            else:
                action = np.argmax(Q_table[state[0], state[1]]) # greedy action with probability 1 - epsilon

            next_state, reward = step(state, action)
            total_rewards += reward # update total_rewards to reflect this step's reward

            # Q-learning update
            best_next_q = np.max(Q_table[next_state[0], next_state[1]]) # optimistic update compared to SARSA's realistic update
            Q_table[state[0], state[1], action] += alpha * (reward + gamma * best_next_q - Q_table[state[0], state[1], action])

            state = next_state
        e_rewards.append(total_rewards) # append total_rewards for this episode to the e_rewards list for plotting
    return Q_table, e_rewards

num_episodes = 300
num_runs = 5
qall_rewards = []
q_trained_table = None
for e in range(num_runs): # 5 independent runs of q-learning
    Q_table, rewards = q_learning(episodes=num_episodes)
    qall_rewards.append(rewards)
    q_trained_table = np.copy(Q_table)

qall_rewards = np.array(qall_rewards)

plt.figure(figsize=(10, 6))
episodes = np.arange(num_episodes)
# plot reward per episode
for i in range(num_runs):
    plt.plot(episodes, qall_rewards[i], alpha=0.4, label=f'Run {i+1}')

# mean and variance calculation
qmean_rewards = np.mean(qall_rewards, axis=0)
qstd_rewards = np.std(qall_rewards, axis=0)
# plot mean
plt.plot(episodes, qmean_rewards, label='Mean Reward', color='black', linewidth=2)
# plot variance
plt.fill_between(episodes, qmean_rewards - qstd_rewards, qmean_rewards + qstd_rewards, color='black', alpha=0.3, label='Variance') # shaded area for variance

plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Q-Learning Performance')
plt.legend()
plt.show()

# Task 3: SARSA
def sarsa(episodes = 200, alpha=0.1, epsilon=0.1):
    Q_table = np.zeros((GRID_ROWS, GRID_COLS, 4))
    e_rewards = [] # track rewards per episode for plotting
    for e in range(episodes): # iterate over number of episodes
        state = START_STATE
        total_rewards = 0
        # epsilon-greedy action selection for initial action
        if np.random.uniform(0, 1) < epsilon:
            action = np.random.randint(0, 4)
        else:
            action = np.argmax(Q_table[state[0], state[1]])
        for step_idx in range(500): # iterate over max 500 steps per episode
            if state == GOAL_STATE:
                break

            next_state, reward = step(state, action)
            total_rewards += reward # update total_rewards to reflect this step's reward

            # epsilon-greedy action selection for next action (on-policy)
            if np.random.uniform(0, 1) < epsilon:
                next_action = np.random.randint(0, 4)
            else:
                next_action = np.argmax(Q_table[next_state[0], next_state[1]])

            # SARSA update (uses realistic next_action instead of best_next_q in Q-learning)
            Q_table[state[0], state[1], action] += alpha * (reward + gamma * Q_table[next_state[0], next_state[1], next_action] - Q_table[state[0], state[1], action])

            state = next_state # update state to next_state for the next iteration
            action = next_action # update action to next_action for the next iteration
        e_rewards.append(total_rewards) # append total_rewards for this episode to the e_rewards list for plotting
    return Q_table, e_rewards

num_episodes = 300
num_runs = 5
sall_rewards = []
s_trained_table = None
for e in range(num_runs): # 5 independent runs of sarsa
    Q_table, rewards = sarsa(episodes=num_episodes)
    sall_rewards.append(rewards)
    s_trained_table = np.copy(Q_table)

sall_rewards = np.array(sall_rewards)

plt.figure(figsize=(10, 6))
episodes = np.arange(num_episodes)
# plot reward per episode
for i in range(num_runs):
    plt.plot(episodes, sall_rewards[i], alpha=0.4, label=f'Run {i+1}')

# mean and variance calculation
smean_rewards = np.mean(sall_rewards, axis=0)
sstd_rewards = np.std(sall_rewards, axis=0)
# plot mean
plt.plot(episodes, smean_rewards, label='Mean Reward', color='black', linewidth=2)
# plot variance
plt.fill_between(episodes, smean_rewards - sstd_rewards, smean_rewards + sstd_rewards, color='black', alpha=0.3, label='Variance') # shaded area for variance

plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('SARSA Performance')
plt.legend()
plt.show()

# Task 4: Algorithm Comparison
plt.figure(figsize=(10, 6))
episodes = np.arange(num_episodes)
# plot Q-learning rewards
plt.plot(episodes, qmean_rewards, label='Q-Learning Mean Reward', color='blue', linewidth=2)
# plot Q-learning variance
plt.fill_between(episodes, qmean_rewards - qstd_rewards, qmean_rewards + qstd_rewards, color='blue', alpha=0.2, label='Q-Learning Variance') # shaded area for variance
# plot SARSA rewards
plt.plot(episodes, smean_rewards, label='SARSA Mean Reward', color='red', linewidth=2)
# plot SARSA variance
plt.fill_between(episodes, smean_rewards - sstd_rewards, smean_rewards + sstd_rewards, color='red', alpha=0.2, label='SARSA Variance') # shaded area for variance
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Q-Learning vs SARSA Performance')
plt.legend()
plt.show()

# Task 5: Value Functions and Policies
q_value = np.max(q_trained_table, axis=2)
q_policy = np.argmax(q_trained_table, axis=2)
s_value = np.max(s_trained_table, axis=2)
s_policy = np.argmax(s_trained_table, axis=2)

q_value[GOAL_STATE] = 0
s_value[GOAL_STATE] = 0

fig, axes = plt.subplots(1, 2, figsize=(20, 6))
axes[0].imshow(q_value, cmap='viridis')
axes[0].set_title('Q-Learning: Value Function and Policy')
axes[0].set_xticks(np.arange(GRID_COLS))
axes[0].set_yticks(np.arange(GRID_ROWS))
axes[0].set_xticks(np.arange(-0.5, GRID_COLS, 1), minor=True)
axes[0].set_yticks(np.arange(-0.5, GRID_ROWS, 1), minor=True)
axes[0].grid(which='minor', color='black', linestyle='-', linewidth=1)
axes[0].tick_params(which='minor', bottom=False, left=False)

for i in range(GRID_ROWS):
    for j in range(GRID_COLS):
        state = (i, j)
        if state in CLIFF_STATES:
            axes[0].text(j, i, 'C', ha='center', va='center', color='red', fontsize=14, weight='bold')
        elif state == START_STATE:
            axes[0].text(j, i, 'S', ha='center', va='center', color='white', fontsize=14, weight='bold')
        elif state == GOAL_STATE:
            axes[0].text(j, i, 'G', ha='center', va='center', color='white', fontsize=14, weight='bold')
        else:
            axes[0].text(j, i, f'{q_value[i, j]:.1f}', ha='center', va='center', color='white', fontsize=9)
            action = q_policy[i, j]
            dx, dy = 0, 0
            if action == 0:
                dy = -0.3
            elif action == 1:
                dx = 0.3
            elif action == 2:
                dy = 0.3
            elif action == 3:
                dx = -0.3
            axes[0].arrow(j, i, dx, dy, head_width=0.12, head_length=0.10, fc='white', ec='white')

fig.colorbar(axes[0].images[0], ax=axes[0], fraction=0.046, pad=0.04)
axes[1].imshow(s_value, cmap='viridis')
axes[1].set_title('SARSA: Value Function and Policy')
axes[1].set_xticks(np.arange(GRID_COLS))
axes[1].set_yticks(np.arange(GRID_ROWS))
axes[1].set_xticks(np.arange(-0.5, GRID_COLS, 1), minor=True)
axes[1].set_yticks(np.arange(-0.5, GRID_ROWS, 1), minor=True)
axes[1].grid(which='minor', color='black', linestyle='-', linewidth=1)
axes[1].tick_params(which='minor', bottom=False, left=False)

for i in range(GRID_ROWS):
    for j in range(GRID_COLS):
        state = (i, j)
        if state in CLIFF_STATES:
            axes[1].text(j, i, 'C', ha='center', va='center', color='red', fontsize=14, weight='bold')
        elif state == START_STATE:
            axes[1].text(j, i, 'S', ha='center', va='center', color='white', fontsize=14, weight='bold')
        elif state == GOAL_STATE:
            axes[1].text(j, i, 'G', ha='center', va='center', color='white', fontsize=14, weight='bold')
        else:
            axes[1].text(j, i, f'{s_value[i, j]:.1f}', ha='center', va='center', color='white', fontsize=9)
            action = s_policy[i, j]
            dx, dy = 0, 0
            if action == 0:
                dy = -0.3
            elif action == 1:
                dx = 0.3
            elif action == 2:
                dy = 0.3
            elif action == 3:
                dx = -0.3
            axes[1].arrow(j, i, dx, dy, head_width=0.12, head_length=0.10, fc='white', ec='white')

fig.colorbar(axes[1].images[0], ax=axes[1], fraction=0.046, pad=0.04)
fig.tight_layout()
plt.show()
