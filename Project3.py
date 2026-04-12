import numpy as np

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
