import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sps

# Plot for shape = 4.0 and scale = 5.0
fig, (axes_0, axes_1, axes_2) = plt.subplots(nrows=1, ncols=3, figsize=(10,5)) # To plot 3 distributions in one row
shape_1 = 4.0 # Shape and Scale for first distribution
scale_1 = 5.0
samples_1 = np.random.gamma(shape_1, scale_1, 1000)
count_1, bins_1, ignored_1 = axes_0.hist(samples_1, 50, density=True, alpha=0.6) # Plot the first histogram
y_1 = bins_1**(shape_1-1) * (np.exp(-bins_1/scale_1) / (sps.gamma(shape_1) * scale_1**shape_1))
axes_0.plot(bins_1, y_1, linewidth=2, color='r', label='Theoretical PDF: (4.0, 5.0)') # Plotting the first theoretical PDF
mean_1 = shape_1 * scale_1
axes_0.scatter(mean_1, 0, linewidth=2, color="k",label=f"Mean: {mean_1}") # To plot the mean of the distribution as one point
axes_0.set_xlabel("Bins")
axes_0.set_ylabel("Density")
axes_0.set_title("Arm0")
axes_0.legend()

# Plot for shape = 25.0 and scale = 2.0
shape_2 = 25.0 # Shape and scale for second distribution
scale_2 = 2.0
samples_2 = np.random.gamma(shape_2, scale_2, 1000)
count_2, bins_2, ignored_2 = axes_1.hist(samples_2, 50, density=True, alpha=0.6) # Plot the second histogram
y_2 = bins_2**(shape_2-1) * (np.exp(-bins_2/scale_2) / (sps.gamma(shape_2) * scale_2**shape_2))
axes_1.plot(bins_2, y_2, linewidth=2, color='g', label='Theoretical PDF: (25.0, 2.0)') # Plot the second theoretical PDF
mean_2 = shape_2 * scale_2
axes_1.scatter(mean_2, 0, linewidth=2, color="k",label=f"Mean: {mean_2} *optimal arm*") # Plot the mean of this distribution
axes_1.set_xlabel("Bins")
axes_1.set_ylabel("Density")
axes_1.set_title("Gamma Distributions and PDFs\nArm1")
axes_1.legend()

# Plot for shape = 1.0 and scale = 8.0
shape_3 = 1.0 # Shape and scale for third distribution
scale_3 = 8.0
samples_3 = np.random.gamma(shape_3, scale_3, 1000)
count_3, bins_3, ignored_3 = axes_2.hist(samples_3, 50, density=True, alpha=0.6) # plot the third histogram
y_3 = bins_3**(shape_3-1) * (np.exp(-bins_3/scale_3) / (sps.gamma(shape_3) * scale_3**shape_3))
axes_2.plot(bins_3, y_3, linewidth=2, color='b', label='Theoretical PDF: (1.0, 8.0)') # Overlay the theoretical PDF for this distrbution
mean_3 = shape_3 * scale_3
axes_2.scatter(mean_3, 0, linewidth=2, color="k",label=f"Mean: {mean_3}") # Plot the mean of this distribution
axes_2.set_xlabel("Bins")
axes_2.set_ylabel("Density")
axes_2.set_title("Arm2")
axes_2.legend()
plt.tight_layout()
plt.show()

# Greedy Approach (Exploitation vs Exploration)
max_games = 1000 # Max games allowed for this experiment
patience = 100 # Number of similar iterations required before early stopping
eps = [0.10, 0.20, 0.30] # Different epsilon values for the greedy approach, i didn't know what to pick so i just picked these
# The graphs look nice so I think these are good values
tolerance = 1e-1 # For early stopping

# Variables for first Greedy Approach
Q_t1 = np.zeros(3) # Estimated rewards for each arm
N_t1 = np.zeros(3) # Number of times each arm has been pulled
rewards1 = np.zeros(max_games) # Rewards for each iteration
avg_rewards1 = np.zeros(max_games) # Running average for rewards
selection1 = np.zeros(max_games) # Selected arm for each iteration
shapes = [shape_1, shape_2, shape_3]
scales = [scale_1, scale_2, scale_3]
last_game1 = 0
last_game2 = 0
last_game3 = 0
# Greedy Approach with eps[0]
for game in range(max_games): # Iterate over number of games (1000)
    if np.random.rand() < eps[0]: # The greedy approach outlined in the assignment description
        arm = np.random.choice(3) # Random action with probability eps[0]
    else:
        arm = np.argmax(Q_t1) # Greedy action with probability 1-eps[0]
        
    selection1[game] = arm # Keep track of selected arms for each game (for eps[0] case) for optimal action plotting
    reward = np.random.gamma(shapes[arm], scales[arm]) # Sample reward from the selected arm's distribution
    rewards1[game] = reward # Keep track of rewards for each game (for eps[0] case)
    N_t1[arm] += 1 # Update the count for the selected arm (for eps[0] case)
    Q_t1[arm] += (reward - Q_t1[arm]) / N_t1[arm] # Update estimated reward for selected arm (for eps[0] case)
    sum_rewards = np.sum(rewards1[:game+1]) # Sum rewards up to the current game (for eps[0] case)
    avg_rewards1[game] = sum_rewards / (game + 1) # Update running average reward (for eps[0] case)
    if (game > 100 and (np.abs(avg_rewards1[game-100] - avg_rewards1[game]) < tolerance)): # Early Stopping Criteria
        last_game1 = game
        break
    elif(game == max_games-1): # If agent reaches max games, set last game number to 1000 for display purposes
        last_game1 = max_games - 1
    
# Variables for second Greedy Approach
Q_t2 = np.zeros(3)
N_t2 = np.zeros(3)
rewards2 = np.zeros(max_games)
avg_rewards2 = np.zeros(max_games)
selection2 = np.zeros(max_games)
# Greedy Approach with eps[1]
for game in range(max_games): # Same process as the first greedy approach, but with a different epsilon value and variables for tracking/plotting
    if np.random.rand() < eps[1]: # Apologies, I didn't want to go through the trouble to define a function.
        arm = np.random.choice(3)
    else:
        arm = np.argmax(Q_t2)
    
    selection2[game] = arm
    reward = np.random.gamma(shapes[arm], scales[arm])
    rewards2[game] = reward
    N_t2[arm] += 1
    Q_t2[arm] += (reward - Q_t2[arm]) / N_t2[arm]
    sum_rewards = np.sum(rewards2[:game+1]) 
    avg_rewards2[game] = sum_rewards / (game + 1)
    if (game > 100 and (np.abs(avg_rewards2[game-100] - avg_rewards2[game]) < tolerance)):
        last_game2 = game
        break
    elif(game == max_games-1):
        last_game2 = max_games - 1
        
# Variables for third Greedy Approach
Q_t3 = np.zeros(3)
N_t3 = np.zeros(3)
rewards3 = np.zeros(max_games)
avg_rewards3 = np.zeros(max_games)
selection3 = np.zeros(max_games)
# Greedy Approach with eps[2]
for game in range(max_games): # Same process as the first greedy approach, but with a different epsilon value and variables for tracking/plotting
    if np.random.rand() < eps[2]: # Apologies, I didn't want to go through the trouble to define a function.
        arm = np.random.choice(3)
    else:
        arm = np.argmax(Q_t3)
        
    selection3[game] = arm
    reward = np.random.gamma(shapes[arm], scales[arm])
    rewards3[game] = reward
    N_t3[arm] += 1
    Q_t3[arm] += (reward - Q_t3[arm]) / N_t3[arm]
    sum_rewards = np.sum(rewards3[:game+1]) 
    avg_rewards3[game] = sum_rewards / (game + 1)
    if game > 100 and (np.abs(avg_rewards3[game-100] - avg_rewards3[game]) < tolerance):
        last_game3 = game
        break
    elif(game == max_games-1):
        last_game3 = max_games -1
        
# Upper Confidence Bound (UCB) Approach
Q_ucb = np.zeros(3)
N_ucb = np.zeros(3)
arms = np.zeros(3)
selection_ucb = np.zeros(max_games)
rewards_ucb = np.zeros(max_games)
avg_rewards_ucb = np.zeros(max_games)
last_game_ucb = 0
# Pull each arm once for initialization (to get rid of divide by 0 warnings)
for arm in range(3):
    reward = np.random.gamma(shapes[arm], scales[arm])
    rewards_ucb[arm] = reward
    N_ucb[arm] += 1
    Q_ucb[arm] += (reward - Q_ucb[arm]) / N_ucb[arm]
    sum_rewards = np.sum(rewards_ucb[:arm+1]) 
    avg_rewards_ucb[arm] = sum_rewards / (arm + 1)
    selection_ucb[arm] = arm
# Iterate over the number of games
for game in range(3, max_games):
    for arm in range(3): # Calculate the arm with the highest UCB value
        ucb_value = Q_ucb[arm] + 2*np.sqrt((np.log(game + 1)) / (N_ucb[arm])) # UCB formula from slides and assignment description
        if arm == 0:
            arms[0] = ucb_value
        elif arm == 1:
            arms[1] = ucb_value
        else:
            arms[2] = ucb_value
    arm = np.argmax(arms) # Select the arm with the highest UCB value
    selection_ucb[game] = arm # Keep track of selected arms for each game for optimal action plotting
    reward = np.random.gamma(shapes[arm], scales[arm]) # Sample reward from the selected arm's distribution
    rewards_ucb[game] = reward # Keep track of rewards for each game
    N_ucb[arm] += 1 # Update the count for the selected arm
    Q_ucb[arm] += (1.0/N_ucb[arm]) * (reward - Q_ucb[arm]) # Update estimated reward for selected arm
    sum_rewards = np.sum(rewards_ucb[:game+1]) # Sum rewards up to the current game
    avg_rewards_ucb[game] = sum_rewards / (game + 1) # Update running average reward
    if game > 100 and (np.abs(avg_rewards_ucb[game-100] - avg_rewards_ucb[game]) < tolerance): # Early Stopping Criteria
        last_game_ucb = game
        break
    elif(game == max_games-1): # If agent reaches max games, set last game number to 1000 for display purposes
        last_game_ucb = max_games -1
    
print(f"Last game for epsilon = {eps[0]}: {last_game1} | Average Reward: {avg_rewards1[last_game1]}") # Excuse my print debugging, it does show useful stuff though
print(f"Last game for epsilon = {eps[1]}: {last_game2} | Average Reward: {avg_rewards2[last_game2]}")
print(f"Last game for epsilon = {eps[2]}: {last_game3} | Average Reward: {avg_rewards3[last_game3]}")
print(f"Last game for UCB:           {last_game_ucb} | Average Reward: {avg_rewards_ucb[last_game_ucb]}")

avg_rewards1[last_game1:] = avg_rewards1[last_game1] # Assign all values after the last game to the last valid game's value
avg_rewards2[last_game2:] = avg_rewards2[last_game2]
avg_rewards3[last_game3:] = avg_rewards3[last_game3]
avg_rewards_ucb[last_game_ucb:] = avg_rewards_ucb[last_game_ucb]

# Plotting average rewards
plt.figure(figsize=(10, 5))
plt.plot(avg_rewards1, label=f"epsilon = {eps[0]}") # Plot average rewards for each approach
plt.plot(avg_rewards2, label=f"epsilon = {eps[1]}")
plt.plot(avg_rewards3, label=f"epsilon = {eps[2]}")
plt.plot(avg_rewards_ucb, label="UCB")
plt.xlim(0, max(last_game1, last_game2, last_game3)) # crop x axis (num games) for display purposes, nicer looking graph
plt.xlabel("Games")
plt.ylabel("Average Reward")
plt.legend()
plt.title("Average Rewards Over Time")
plt.show()

# Plotting optimal action selection
optimal_arm = 1 # Index 1: Second arm is optimal (mean reward of 50)
percent_selection1 = np.zeros(max_games) # Initialize arrays to store % optimal action selection for plotting each approach
percent_selection2 = np.zeros(max_games)
percent_selection3 = np.zeros(max_games)
percent_selection_ucb = np.zeros(max_games)

for i in range(max_games):
    percent_selection1[i] = (np.sum(selection1[:i+1] == optimal_arm) / (i + 1)) * 100 # Calculate % optimal action for each approach at each time step (each game)
    percent_selection2[i] = (np.sum(selection2[:i+1] == optimal_arm) / (i + 1)) * 100
    percent_selection3[i] = (np.sum(selection3[:i+1] == optimal_arm) / (i + 1)) * 100
    percent_selection_ucb[i] = (np.sum(selection_ucb[:i+1] == optimal_arm) / (i + 1)) * 100
    
percent_selection1[last_game1:] = percent_selection1[last_game1-1] # Assign all values after the last game to the last valid game's value
percent_selection2[last_game2:] = percent_selection2[last_game2-1]
percent_selection3[last_game3:] = percent_selection3[last_game3-1]
percent_selection_ucb[last_game_ucb:] = percent_selection_ucb[last_game_ucb-1]

# Plotting % optimal action for all experiments (greedy approaches and UCB)
plt.figure(figsize=(10, 5))
plt.plot(percent_selection1, label=f"epsilon = {eps[0]}") # Plot % Optimal Action for each approach
plt.plot(percent_selection2, label=f"epsilon = {eps[1]}")
plt.plot(percent_selection3, label=f"epsilon = {eps[2]}")
plt.plot(percent_selection_ucb, label="UCB")
plt.xlim(0, max(last_game1, last_game2, last_game3)) # crop x axis (num games) for display purposes, nicer looking graph
plt.xlabel("Games")
plt.ylabel("% Optimal Action")
plt.legend()
plt.title("Optimal Action Selection Over Time")
plt.show()

import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sps

# Plot for shape = 4.0 and scale = 5.0
fig, (axes_0, axes_1, axes_2) = plt.subplots(nrows=1, ncols=3, figsize=(10,5)) # To plot 3 distributions in one row
shape_1 = 4.0 # Shape and Scale for first distribution
scale_1 = 5.0
samples_1 = np.random.gamma(shape_1, scale_1, 1000)
count_1, bins_1, ignored_1 = axes_0.hist(samples_1, 50, density=True, alpha=0.6) # Plot the first histogram
y_1 = bins_1**(shape_1-1) * (np.exp(-bins_1/scale_1) / (sps.gamma(shape_1) * scale_1**shape_1))
axes_0.plot(bins_1, y_1, linewidth=2, color='r', label='Theoretical PDF: (4.0, 5.0)') # Plotting the first theoretical PDF
mean_1 = shape_1 * scale_1
axes_0.scatter(mean_1, 0, linewidth=2, color="k",label=f"Mean: {mean_1}") # To plot the mean of the distribution as one point
axes_0.set_xlabel("Bins")
axes_0.set_ylabel("Density")
axes_0.set_title("Arm0")
axes_0.legend()

# Plot for shape = 25.0 and scale = 2.0
shape_2 = 25.0 # Shape and scale for second distribution
scale_2 = 2.0
samples_2 = np.random.gamma(shape_2, scale_2, 1000)
count_2, bins_2, ignored_2 = axes_1.hist(samples_2, 50, density=True, alpha=0.6) # Plot the second histogram
y_2 = bins_2**(shape_2-1) * (np.exp(-bins_2/scale_2) / (sps.gamma(shape_2) * scale_2**shape_2))
axes_1.plot(bins_2, y_2, linewidth=2, color='g', label='Theoretical PDF: (25.0, 2.0)') # Plot the second theoretical PDF
mean_2 = shape_2 * scale_2
axes_1.scatter(mean_2, 0, linewidth=2, color="k",label=f"Mean: {mean_2} *optimal arm*") # Plot the mean of this distribution
axes_1.set_xlabel("Bins")
axes_1.set_ylabel("Density")
axes_1.set_title("Gamma Distributions and PDFs\nArm1")
axes_1.legend()

# Plot for shape = 1.0 and scale = 8.0
shape_3 = 1.0 # Shape and scale for third distribution
scale_3 = 8.0
samples_3 = np.random.gamma(shape_3, scale_3, 1000)
count_3, bins_3, ignored_3 = axes_2.hist(samples_3, 50, density=True, alpha=0.6) # plot the third histogram
y_3 = bins_3**(shape_3-1) * (np.exp(-bins_3/scale_3) / (sps.gamma(shape_3) * scale_3**shape_3))
axes_2.plot(bins_3, y_3, linewidth=2, color='b', label='Theoretical PDF: (1.0, 8.0)') # Overlay the theoretical PDF for this distrbution
mean_3 = shape_3 * scale_3
axes_2.scatter(mean_3, 0, linewidth=2, color="k",label=f"Mean: {mean_3}") # Plot the mean of this distribution
axes_2.set_xlabel("Bins")
axes_2.set_ylabel("Density")
axes_2.set_title("Arm2")
axes_2.legend()
plt.tight_layout()
plt.show()

# Greedy Approach (Exploitation vs Exploration)
max_games = 1000 # Max games allowed for this experiment
patience = 100 # Number of similar iterations required before early stopping
eps = [0.10, 0.20, 0.30] # Different epsilon values for the greedy approach, i didn't know what to pick so i just picked these
# The graphs look nice so I think these are good values
tolerance = 1e-1 # For early stopping

# Variables for first Greedy Approach
Q_t1 = np.zeros(3) # Estimated rewards for each arm
N_t1 = np.zeros(3) # Number of times each arm has been pulled
rewards1 = np.zeros(max_games) # Rewards for each iteration
avg_rewards1 = np.zeros(max_games) # Running average for rewards
selection1 = np.zeros(max_games) # Selected arm for each iteration
shapes = [shape_1, shape_2, shape_3]
scales = [scale_1, scale_2, scale_3]
last_game1 = 0
last_game2 = 0
last_game3 = 0
# Greedy Approach with eps[0]
for game in range(max_games): # Iterate over number of games (1000)
    if np.random.rand() < eps[0]: # The greedy approach outlined in the assignment description
        arm = np.random.choice(3) # Random action with probability eps[0]
    else:
        arm = np.argmax(Q_t1) # Greedy action with probability 1-eps[0]
        
    selection1[game] = arm # Keep track of selected arms for each game (for eps[0] case) for optimal action plotting
    reward = np.random.gamma(shapes[arm], scales[arm]) # Sample reward from the selected arm's distribution
    rewards1[game] = reward # Keep track of rewards for each game (for eps[0] case)
    N_t1[arm] += 1 # Update the count for the selected arm (for eps[0] case)
    Q_t1[arm] += (reward - Q_t1[arm]) / N_t1[arm] # Update estimated reward for selected arm (for eps[0] case)
    sum_rewards = np.sum(rewards1[:game+1]) # Sum rewards up to the current game (for eps[0] case)
    avg_rewards1[game] = sum_rewards / (game + 1) # Update running average reward (for eps[0] case)
    if (game > 100 and (np.abs(avg_rewards1[game-100] - avg_rewards1[game]) < tolerance)): # Early Stopping Criteria
        last_game1 = game
        break
    elif(game == max_games-1): # If agent reaches max games, set last game number to 1000 for display purposes
        last_game1 = max_games - 1
    
# Variables for second Greedy Approach
Q_t2 = np.zeros(3)
N_t2 = np.zeros(3)
rewards2 = np.zeros(max_games)
avg_rewards2 = np.zeros(max_games)
selection2 = np.zeros(max_games)
# Greedy Approach with eps[1]
for game in range(max_games): # Same process as the first greedy approach, but with a different epsilon value and variables for tracking/plotting
    if np.random.rand() < eps[1]: # Apologies, I didn't want to go through the trouble to define a function.
        arm = np.random.choice(3)
    else:
        arm = np.argmax(Q_t2)
    
    selection2[game] = arm
    reward = np.random.gamma(shapes[arm], scales[arm])
    rewards2[game] = reward
    N_t2[arm] += 1
    Q_t2[arm] += (reward - Q_t2[arm]) / N_t2[arm]
    sum_rewards = np.sum(rewards2[:game+1]) 
    avg_rewards2[game] = sum_rewards / (game + 1)
    if (game > 100 and (np.abs(avg_rewards2[game-100] - avg_rewards2[game]) < tolerance)):
        last_game2 = game
        break
    elif(game == max_games-1):
        last_game2 = max_games - 1
        
# Variables for third Greedy Approach
Q_t3 = np.zeros(3)
N_t3 = np.zeros(3)
rewards3 = np.zeros(max_games)
avg_rewards3 = np.zeros(max_games)
selection3 = np.zeros(max_games)
# Greedy Approach with eps[2]
for game in range(max_games): # Same process as the first greedy approach, but with a different epsilon value and variables for tracking/plotting
    if np.random.rand() < eps[2]: # Apologies, I didn't want to go through the trouble to define a function.
        arm = np.random.choice(3)
    else:
        arm = np.argmax(Q_t3)
        
    selection3[game] = arm
    reward = np.random.gamma(shapes[arm], scales[arm])
    rewards3[game] = reward
    N_t3[arm] += 1
    Q_t3[arm] += (reward - Q_t3[arm]) / N_t3[arm]
    sum_rewards = np.sum(rewards3[:game+1]) 
    avg_rewards3[game] = sum_rewards / (game + 1)
    if game > 100 and (np.abs(avg_rewards3[game-100] - avg_rewards3[game]) < tolerance):
        last_game3 = game
        break
    elif(game == max_games-1):
        last_game3 = max_games -1
        
# Upper Confidence Bound (UCB) Approach
Q_ucb = np.zeros(3)
N_ucb = np.zeros(3)
arms = np.zeros(3)
selection_ucb = np.zeros(max_games)
rewards_ucb = np.zeros(max_games)
avg_rewards_ucb = np.zeros(max_games)
last_game_ucb = 0
# Pull each arm once for initialization (to get rid of divide by 0 warnings)
for arm in range(3):
    reward = np.random.gamma(shapes[arm], scales[arm])
    rewards_ucb[arm] = reward
    N_ucb[arm] += 1
    Q_ucb[arm] += (reward - Q_ucb[arm]) / N_ucb[arm]
    sum_rewards = np.sum(rewards_ucb[:arm+1]) 
    avg_rewards_ucb[arm] = sum_rewards / (arm + 1)
    selection_ucb[arm] = arm
# Iterate over the number of games
for game in range(3, max_games):
    for arm in range(3): # Calculate the arm with the highest UCB value
        ucb_value = Q_ucb[arm] + 2*np.sqrt((np.log(game + 1)) / (N_ucb[arm])) # UCB formula from slides and assignment description
        if arm == 0:
            arms[0] = ucb_value
        elif arm == 1:
            arms[1] = ucb_value
        else:
            arms[2] = ucb_value
    arm = np.argmax(arms) # Select the arm with the highest UCB value
    selection_ucb[game] = arm # Keep track of selected arms for each game for optimal action plotting
    reward = np.random.gamma(shapes[arm], scales[arm]) # Sample reward from the selected arm's distribution
    rewards_ucb[game] = reward # Keep track of rewards for each game
    N_ucb[arm] += 1 # Update the count for the selected arm
    Q_ucb[arm] += (1.0/N_ucb[arm]) * (reward - Q_ucb[arm]) # Update estimated reward for selected arm
    sum_rewards = np.sum(rewards_ucb[:game+1]) # Sum rewards up to the current game
    avg_rewards_ucb[game] = sum_rewards / (game + 1) # Update running average reward
    if game > 100 and (np.abs(avg_rewards_ucb[game-100] - avg_rewards_ucb[game]) < tolerance): # Early Stopping Criteria
        last_game_ucb = game
        break
    elif(game == max_games-1): # If agent reaches max games, set last game number to 1000 for display purposes
        last_game_ucb = max_games -1
    
print(f"Last game for epsilon = {eps[0]}: {last_game1} | Average Reward: {avg_rewards1[last_game1]}") # Excuse my print debugging, it does show useful stuff though
print(f"Last game for epsilon = {eps[1]}: {last_game2} | Average Reward: {avg_rewards2[last_game2]}")
print(f"Last game for epsilon = {eps[2]}: {last_game3} | Average Reward: {avg_rewards3[last_game3]}")
print(f"Last game for UCB:           {last_game_ucb} | Average Reward: {avg_rewards_ucb[last_game_ucb]}")

avg_rewards1[last_game1:] = avg_rewards1[last_game1] # Assign all values after the last game to the last valid game's value
avg_rewards2[last_game2:] = avg_rewards2[last_game2]
avg_rewards3[last_game3:] = avg_rewards3[last_game3]
avg_rewards_ucb[last_game_ucb:] = avg_rewards_ucb[last_game_ucb]

# Plotting average rewards
plt.figure(figsize=(10, 5))
plt.plot(avg_rewards1, label=f"epsilon = {eps[0]}") # Plot average rewards for each approach
plt.plot(avg_rewards2, label=f"epsilon = {eps[1]}")
plt.plot(avg_rewards3, label=f"epsilon = {eps[2]}")
plt.plot(avg_rewards_ucb, label="UCB")
plt.xlim(0, max(last_game1, last_game2, last_game3)) # crop x axis (num games) for display purposes, nicer looking graph
plt.xlabel("Games")
plt.ylabel("Average Reward")
plt.legend()
plt.title("Average Rewards Over Time")
plt.show()

# Plotting optimal action selection
optimal_arm = 1 # Index 1: Second arm is optimal (mean reward of 50)
percent_selection1 = np.zeros(max_games) # Initialize arrays to store % optimal action selection for plotting each approach
percent_selection2 = np.zeros(max_games)
percent_selection3 = np.zeros(max_games)
percent_selection_ucb = np.zeros(max_games)

for i in range(max_games):
    percent_selection1[i] = (np.sum(selection1[:i+1] == optimal_arm) / (i + 1)) * 100 # Calculate % optimal action for each approach at each time step (each game)
    percent_selection2[i] = (np.sum(selection2[:i+1] == optimal_arm) / (i + 1)) * 100
    percent_selection3[i] = (np.sum(selection3[:i+1] == optimal_arm) / (i + 1)) * 100
    percent_selection_ucb[i] = (np.sum(selection_ucb[:i+1] == optimal_arm) / (i + 1)) * 100
    
percent_selection1[last_game1:] = percent_selection1[last_game1-1] # Assign all values after the last game to the last valid game's value
percent_selection2[last_game2:] = percent_selection2[last_game2-1]
percent_selection3[last_game3:] = percent_selection3[last_game3-1]
percent_selection_ucb[last_game_ucb:] = percent_selection_ucb[last_game_ucb-1]

# Plotting % optimal action for all experiments (greedy approaches and UCB)
plt.figure(figsize=(10, 5))
plt.plot(percent_selection1, label=f"epsilon = {eps[0]}") # Plot % Optimal Action for each approach
plt.plot(percent_selection2, label=f"epsilon = {eps[1]}")
plt.plot(percent_selection3, label=f"epsilon = {eps[2]}")
plt.plot(percent_selection_ucb, label="UCB")
plt.xlim(0, max(last_game1, last_game2, last_game3)) # crop x axis (num games) for display purposes, nicer looking graph
plt.xlabel("Games")
plt.ylabel("% Optimal Action")
plt.legend()
plt.title("Optimal Action Selection Over Time")
plt.show()
