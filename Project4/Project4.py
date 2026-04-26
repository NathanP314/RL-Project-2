import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import mse_loss
from torch.distributions import Normal
import gymnasium as gym

from buffer import Buffer, collect_data, act, rescale_actions
from vpg import _log_prob, build_actor, train_vpg
from Modules import NormalModule
from plotting import plot_learning_curves, plot_loss_curves
from video import record_video, generate_strobe
from gae import build_critic, compute_gae

"""
HW4 — Task 1: Replay buffer and environment interaction.

Complete the four TODO items below before moving on to vpg.py.
"""

class Buffer:
    """Experience replay buffer storing one-step transitions.

    Use-contract:
        add(state, action, reward, done)             — push one transition
        calc_reward_to_go(gamma)                     — fill self.ret_to_go
        sample(batch_size) -> tuple of numpy arrays  — draw a mini-batch
    """

    def __init__(self, sdim, adim, size, sdtype=np.float32, adtype=np.float32, ep_len=200):
        self.states    = np.zeros((size, sdim), dtype=sdtype)
        self.actions   = np.zeros((size, adim), dtype=adtype)
        self.rewards   = np.zeros((size, 1),    dtype=np.float32)
        self.ret_to_go = np.zeros((size, 1),    dtype=np.float32)
        self.dones     = np.zeros((size, 1),    dtype=bool)
        self.i     = 0
        self.size  = size
        self.max_i = 0
        self.ep_len = ep_len

    def add(self, state, action, reward, done):
        self.states[self.i] = state
        self.actions[self.i] = action
        self.rewards[self.i] = reward
        self.dones[self.i] = done
        self.i = (self.i + 1) % self.size
        self.max_i = min(self.max_i + 1, self.size) 

    def sample(self, batch_size):
        upper = max(self.max_i - 1, 1)
        idxs = np.random.randint(0, upper, size=batch_size)
        done_mask = self.dones[idxs, 0]
        idxs = np.where(done_mask, np.maximum(idxs - 1, 0), idxs)
        next_idxs = idxs + 1
        return (
            self.states[idxs],
            self.actions[idxs],
            self.rewards[idxs],
            self.states[next_idxs],
            self.dones[next_idxs],
            self.ret_to_go[idxs],
            self.ret_to_go[next_idxs],
        )

    def calc_reward_to_go(self, gamma=0.975):
        ret = 0
        for i in reversed(range(self.max_i)):
            if self.dones[i, 0]:
                ret = 0
            ret = self.rewards[i, 0] + gamma * ret
            self.ret_to_go[i, 0] = ret


def collect_data(size, env, agent, title="collecting"):
    """Roll out `agent` (a policy network) in `env` for `size` steps.

    Returns:
        buffer  — a populated Buffer
        avg_rwd — average per-step reward observed during the rollout
    """
    s, _ = env.reset()
    buffer = Buffer(sdim=len(s), adim=1, size=size)
    cumulative_reward = 0
    episode_rewards = []
    episode_reward = 0
    for _ in range(size):
        a = act(policy=agent, state=s)
        a_scaled = rescale_actions(a, -2.0, 2.0)
        s2, r, terminated, truncated, _ = env.step(a_scaled)
        done = truncated or terminated
        buffer.add(state=s, action=a_scaled, reward=r, done=done)
        cumulative_reward += r
        episode_reward += r
        s = s2
        if done:
            episode_rewards.append(episode_reward)
            episode_reward = 0
            s, _ = env.reset()
    buffer.calc_reward_to_go()
    avg_step_reward = cumulative_reward / size
    if episode_rewards:
        avg_episode_reward = np.mean(episode_rewards)
    else:
        avg_episode_reward = 0
    return buffer, avg_step_reward, avg_episode_reward

def act(policy, state):
    """Sample a continuous action a ~ N(mu(state), sigma) from the policy."""
    state_tensor = torch.tensor(state, dtype=torch.float32)
    mu, log_std = policy(state_tensor)
    std = torch.exp(log_std)
    distribution = Normal(mu, std)
    action = distribution.sample()
    return action.detach().numpy().reshape(-1)
    


def rescale_actions(action, amin, amax):
    """Rescale a tanh-squashed action from (-1, 1) to the env range [amin, amax]."""
    return amin + (action + 1.0) * 0.5 * (amax-amin)

"""
HW4 — Task 2: Vanilla policy gradient (REINFORCE).

Depends on: buffer.py (Task 1 must be complete).
"""


# ---------------------------------------------------------------------------
# Shared helpers (provided — do not modify)
# ---------------------------------------------------------------------------

def _log_prob(policy, states, actions):
    """Compute sum of log-probabilities under the current policy."""
    mu, sigma = policy(states)
    return Normal(mu, sigma).log_prob(actions).sum(dim=-1, keepdim=True)


def build_actor(state_dim, action_dim, hidden_size):
    """Two-layer feed-forward actor ending in NormalModule (provided).

    Architecture:
        Linear(state_dim, hidden_size) -> ReLU
        -> Linear(hidden_size, hidden_size) -> ReLU
        -> NormalModule(hidden_size, action_dim)
    """
    return nn.Sequential(
        nn.Linear(state_dim, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, hidden_size),
        nn.ReLU(),
        NormalModule(hidden_size, action_dim),
    )


# ---------------------------------------------------------------------------
# Task 2 TODOs
# ---------------------------------------------------------------------------

def reinforce_signal(policy, states, actions, rewards_to_go, avg_rwd, use_avg=False):
    """Vanilla policy-gradient loss weighted by reward-to-go."""
    # TODO: compute  -E[ (R_to_go - baseline?) * log pi(a | s) ]
    log_probs = _log_prob(policy=policy, states=states, actions=actions)
    if use_avg:
        advantage = rewards_to_go - avg_rwd
    else:
        advantage = rewards_to_go
    loss = -(advantage * log_probs).mean()
    return loss


def reinforce_rwd_signal(policy, states, actions, rewards):
    """REINFORCE loss using one-step rewards instead of reward-to-go."""
    # TODO: compute  -E[ r_t * log pi(a | s) ].
    log_probs = _log_prob(policy=policy, states=states, actions=actions)
    loss = -(rewards * log_probs).mean()
    return loss


def train_vpg(
    epochs=3,
    episodes=10,
    updates=10,
    learning_rate=1e-4,
    hidden_size=32,
    layers=2,
    batch_size=512,
    use_avg=False,
    use_rwds=False,
    gamma=0.975,
):
    """Train the vanilla policy-gradient agent (Task 2).

    Returns:
        policy  — the trained actor network (pass to video.record_video)
        returns — list of per-epoch average episodic returns
    """
    env = gym.make("Pendulum-v1")
    state_dim  = env.reset()[0].shape[0]
    action_dim = env.action_space.sample().shape[0]
    episode_len = env.spec.max_episode_steps

    policy    = build_actor(state_dim, action_dim, hidden_size)
    optimizer = torch.optim.Adam(params=policy.parameters(), lr=learning_rate)

    returns_per_epoch = []
    for x in range(epochs):
        # TODO: 1) roll out to fill a buffer (use collaect_data under torch.no_grad)
        #       2) buffer.calc_reward_to_go()
        with torch.no_grad():
            buffer, avg_step_reward, avg_episode_reward = collect_data(episodes * episode_len, env=env, agent=policy)  # TODO
        buffer.calc_reward_to_go(gamma)
        for i in range(updates):
            # TODO: You need to sample from the buffer here
            # TODO: After sampling you need to convert numpy arrays to tensors, Example: "s_t = torch.as_tensor(s, dtype=torch.float32)"
            states, actions, rewards, _, _, rewards_to_go, _ = buffer.sample(batch_size=batch_size)
            states_tensor = torch.as_tensor(states, dtype=torch.float32)
            actions_tensor = torch.as_tensor(actions, dtype=torch.float32)
            rewards_to_go_tensor = torch.as_tensor(rewards_to_go, dtype=torch.float32)
            rewards_tensor = torch.as_tensor(rewards, dtype=torch.float32)
            
            optimizer.zero_grad()

            # TODO: compute loss here
            if use_rwds:
                loss = reinforce_rwd_signal(policy=policy, states=states_tensor, actions=actions_tensor,rewards=rewards_tensor)
            else:
                loss = reinforce_signal(policy=policy, states=states_tensor, actions=actions_tensor, 
                                        rewards_to_go=rewards_to_go_tensor, avg_rwd=avg_step_reward, use_avg=use_avg)

            loss.backward()
            optimizer.step()

        # TODO: record the epoch's avg episodic return for the learning curve.
        returns_per_epoch.append(avg_episode_reward)
        print(f"Epoch {x+1}: return = {avg_episode_reward:.4f}")
    # TODO: return (policy, list_of_per_epoch_returns).
    return policy, returns_per_epoch
"""
HW4 — Task 3: Critic network and Generalized Advantage Estimation (GAE).

Depends on: buffer.py (Task 1) and vpg.py (Task 2).
"""

# ---------------------------------------------------------------------------
# Shared helper (provided — do not modify)
# ---------------------------------------------------------------------------

def build_critic(state_dim, hidden_size):
    """Two-layer feed-forward critic that outputs a scalar V(s) (provided).

    Architecture:
        Linear(state_dim, hidden_size) -> ReLU
        -> Linear(hidden_size, hidden_size) -> ReLU
        -> Linear(hidden_size, 1)
    """
    return nn.Sequential(
        nn.Linear(state_dim, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, 1),
    )


# ---------------------------------------------------------------------------
# Task 3 TODOs
# ---------------------------------------------------------------------------

def compute_gae(rewards, values, next_values, dones, gamma=0.975, lam=0.95):
    """Generalized Advantage Estimation (PPO paper, Equations 11-12). https://arxiv.org/pdf/1707.06347

        delta_t = r_t + gamma * V(s_{t+1}) * (1 - done_t) - V(s_t)
        A_t     = delta_t + gamma * lam * (1 - done_t) * A_{t+1}
    """
    advantages = np.zeros_like(rewards)
    gae = 0
    for t in reversed(range(len(rewards))):
        boundary = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_values[t] * boundary - values[t]
        gae = delta + gamma * lam * boundary * gae
        advantages[t] = gae
    return advantages

def reinforce_adv_signal(policy, states, actions, advantages):
    """Policy-gradient loss weighted by arbitrary advantages (e.g. GAE)."""
    # TODO: compute  -E[ A_t * log pi(a | s) ].
    log_probs = _log_prob(policy=policy, states=states, actions=actions)
    loss = -(advantages * log_probs).mean()
    return loss


def train_advantage_vpg(
    epochs=3,
    episodes=10,
    updates=10,
    critic_updates=80,
    learning_rate=1e-4,
    critic_lr=3e-4,
    hidden_size=32,
    layers=2,
    batch_size=512,
    gamma=0.975,
    lam=0.95,
):
    """Policy gradient with a learned V(s) baseline and GAE (Task 3).

    Returns:
        policy  — the trained actor network (pass to video.record_video)
        returns — list of per-epoch average episodic returns
    """
    env = gym.make("Pendulum-v1")
    state_dim   = env.reset()[0].shape[0]
    action_dim  = env.action_space.sample().shape[0]
    episode_len = env.spec.max_episode_steps

    policy       = build_actor(state_dim, action_dim, hidden_size)
    critic       = build_critic(state_dim, hidden_size)
    optimizer    = torch.optim.Adam(params=policy.parameters(), lr=learning_rate)
    cr_optimizer = torch.optim.Adam(params=critic.parameters(), lr=critic_lr)

    returns_per_epoch = []
    for x in range(epochs):

        # --- collect experience ---
        with torch.no_grad():
            buffer, avg_step_rwd, avg_episode_reward = collect_data(
                episodes * episode_len, env, policy, title=f"gae {x + 1}/{epochs}"
            )
        
        # TODO: fill buffer.ret_to_go using buffer.calc_reward_to_go(gamma).
        buffer.calc_reward_to_go(gamma)
        
        # --- train the critic ---
        # Regress V(s) toward the reward-to-go targets for critic_updates steps.
        for _ in range(critic_updates):
            states, _, _, _, _, rtg, _ = buffer.sample(batch_size)
            states_t = torch.as_tensor(states, dtype=torch.float32)
            rtg_t    = torch.as_tensor(rtg,    dtype=torch.float32)
            cr_optimizer.zero_grad()
            # TODO: compute mse_loss between critic(states_t) and rtg_t,
            #       then call .backward() and cr_optimizer.step().
            values_preds = critic(states_t)
            loss = mse_loss(values_preds, rtg_t)
            loss.backward()
            cr_optimizer.step()
        # --- compute GAE advantages ---
        # Run the critic (no gradients) on every stored state.
        all_states = torch.as_tensor(buffer.states[: buffer.max_i], dtype=torch.float32)
        with torch.no_grad():
            values = critic(all_states).detach().numpy()          # V(s_t)
        next_values = np.zeros_like(values)
        next_values[:-1] = values[1:]                    # V(s_{t+1}), 0 at episode end

        # TODO: call compute_gae(...) to get an (N, 1) array of advantages.
        advantages = compute_gae(
            rewards=buffer.rewards[:buffer.max_i],
            values=values,
            next_values=next_values,
            dones=buffer.dones[:buffer.max_i],
            gamma=gamma,
            lam=lam
        )  # TODO

        # Normalise for training stability (provided).
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # --- train the actor ---
        for _ in range(updates):
            idxs      = np.random.randint(0, buffer.max_i, size=batch_size)
            states_t  = torch.as_tensor(buffer.states[idxs],  dtype=torch.float32)
            actions_t = torch.as_tensor(buffer.actions[idxs], dtype=torch.float32)
            adv_t     = torch.as_tensor(advantages[idxs],     dtype=torch.float32)
            optimizer.zero_grad()
            # TODO: call reinforce_adv_signal(...) to get the loss,
            #       then call .backward() and optimizer.step().
            loss = reinforce_adv_signal(policy=policy, states=states_t, actions=actions_t, advantages=adv_t)
            loss.backward()
            optimizer.step()
        # ep_return = avg_step_rwd * episode_len
        ep_return = avg_episode_reward
        returns_per_epoch.append(ep_return)
        print(f"gae epoch {x + 1}/{epochs}: return={ep_return:.2f}")

    return policy, returns_per_epoch

    
"""
HW4 — Tasks 4 & 5: PPO surrogate objective and full PPO algorithm.

Depends on: buffer.py (Task 1), vpg.py (Task 2), gae.py (Task 3).
"""

# ---------------------------------------------------------------------------
# Internal helper (provided — do not modify)
# ---------------------------------------------------------------------------

def _critic_values(critic, buffer):
    """Run the critic on every stored state, returning (values, next_values)."""
    states = torch.as_tensor(buffer.states[: buffer.max_i], dtype=torch.float32)
    with torch.no_grad():
        values = critic(states).numpy()
    next_values = np.zeros_like(values)
    next_values[:-1] = values[1:]
    return values, next_values


# ---------------------------------------------------------------------------
# Task 4 TODOs
# ---------------------------------------------------------------------------

def ppo_surrogate_loss(
    policy, states, actions, advantages, old_log_probs, eps_clip=0.2, clip=True
):
    """PPO surrogate objective (PPO paper, Equation 7).

        r_t(theta) = exp( log pi_theta(a|s) - log pi_theta_old(a|s) )

        unclipped:   L = E[ r_t * A_t ]
        clipped:     L = E[ min( r_t * A_t,
                                 clip(r_t, 1 - eps, 1 + eps) * A_t ) ]

    Returns the *negative* of the objective so that optimizer.step()
    performs gradient ascent on the expected return.
    """
    pass


def ppo_total_loss(
    policy,
    critic,
    states,
    actions,
    advantages,
    returns,
    old_log_probs,
    eps_clip=0.2,
    c1=0.5,
    c2=0.01,
    clip=True,
):
    """PPO total loss (PPO paper, Equation 9).

        L_total = L_surr  +  c1 * L_VF  -  c2 * S[pi]

    where L_VF = ( V_theta(s) - R_t )^2  and  S[pi] is the policy entropy.
    Returns a scalar tensor to be minimised.
    """
    pass


# ---------------------------------------------------------------------------
# Task 5 TODO
# ---------------------------------------------------------------------------

def train_ppo(
    iterations=200,
    steps_per_iter=2048,
    sgd_epochs=10,
    minibatch_size=64,
    learning_rate=3e-4,
    hidden_size=64,
    gamma=0.99,
    lam=0.95,
    eps_clip=0.2,
    c1=0.5,
    c2=0.01,
    clip=True,
):
    """Full PPO algorithm with a single actor (N = 1).

    Returns:
        policy  — the trained actor network (pass to video.record_video)
        returns — list of per-iteration average episodic returns
        losses  — list of per-iteration total loss values
    """
    env = gym.make("Pendulum-v1")
    state_dim  = env.reset()[0].shape[0]
    action_dim = env.action_space.sample().shape[0]
    episode_len = env.spec.max_episode_steps

    policy       = build_actor(state_dim, action_dim, hidden_size)
    critic       = build_critic(state_dim, hidden_size)
    optimizer    = torch.optim.Adam(policy.parameters(), lr=learning_rate)
    cr_optimizer = torch.optim.Adam(critic.parameters(), lr=learning_rate)

    returns_per_iter = []
    losses_per_iter  = []

    for k in range(iterations):
        # TODO: 1) roll out the current policy for `steps_per_iter` steps
        #          and store transitions in a Buffer.
        #       2) compute V(s) and V(s') with the critic, then GAE advantages
        #          and target returns (returns = advantages + V(s)).
        #       3) cache the log-probabilities of the sampled actions under
        #          the *old* policy (detach from the graph).
        #       4) for `sgd_epochs` epochs, iterate over minibatches of the
        #          collected data and minimise ppo_total_loss(...).
        #       5) log per-iteration episodic return and total loss for the
        #          required learning / loss curve plots.
        pass

    # TODO: return policy, list_of_returns, list_of_losses

if __name__ == "__main__":
    # Task 1 Random Policy
    env = gym.make("Pendulum-v1")
    policy = build_actor(state_dim=3, action_dim=1, hidden_size=256)
    buffer, avg_step_reward, avg_episode_reward = collect_data(1000, env, policy)
    print(f"AVG STEP REWARD: {avg_step_reward}")
    print(f"AVG EPISODE REWARD: {avg_episode_reward}")
    
    # # Task 2 Training and Plotting
    # policy1, ret1 = train_vpg(epochs=300, learning_rate=1e-3)
    # policy2, ret2 = train_vpg(epochs=300, learning_rate=2e-4)

    # plot_learning_curves(
    # {"lr=1e-3": ret1,"lr=3e-4": ret2,},
    # title="Task 2: rewards-to-go for two learning rates",
    # )
    
    # # Task 3 Training and Plotting
    # policy_rtg, ret_rtg = train_vpg(epochs=500, learning_rate=3e-4)
    # policy_gae, ret_gae = train_advantage_vpg(epochs=500, learning_rate=3e-4)
    # plot_learning_curves(
    #     {"rewards-to-go": ret_rtg, "GAE": ret_gae},
    #     title="Task 3: rewards-to-go vs GAE",
    # )
    
    # record_video(policy_gae, path="videos/task3_gae.mp4")  # optional

    # # Example: compare two learning rates
    # policy_lo, ret_lo = train_vpg(epochs=200, learning_rate=1e-4)
    # policy_hi, ret_hi = train_vpg(epochs=200, learning_rate=3e-4)
    # plot_learning_curves(
    #     {"lr=1e-4": ret_lo, "lr=3e-4": ret_hi},
    #     title="Task 2: VPG with different learning rates",
    # )
    # record_video(policy_hi, path="videos/task2_vpg.mp4")  # optional

    # # --- Task 4: clipped vs unclipped ---
    # _, ret_clip,   loss_clip   = train_ppo(iterations=50, clip=True)
    # _, ret_noclip, loss_noclip = train_ppo(iterations=50, clip=False)
    # plot_learning_curves(
    #     {"clipped": ret_clip, "unclipped": ret_noclip},
    #     title="Task 4: PPO clipped vs unclipped",
    # )
    # plot_loss_curves(
    #     {"clipped": loss_clip, "unclipped": loss_noclip},
    #     title="Task 4: PPO loss curves",
    # )

    # # --- Task 5: full PPO ---
    # policy, ret_ppo, loss_ppo = train_ppo(iterations=500)
    # plot_learning_curves({"PPO": ret_ppo}, title="Task 5: Full PPO")
    # plot_loss_curves({"PPO": loss_ppo}, title="Task 5: Total loss")
    # record_video(policy, path="videos/task5_ppo.mp4")            # optional
    # generate_strobe(policy, path="videos/task5_ppo_strobe.png")  # optional
    