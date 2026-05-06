from collections import deque
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.nn.functional import mse_loss
import gymnasium as gym


# ---------------------------------------------------------------------------
# Plotting from plotting.py
# ---------------------------------------------------------------------------

def _ema(values, decay):
    """Exponential moving average. `decay=0` returns the raw series."""
    values = np.asarray(values, dtype=np.float64)
    if decay <= 0 or len(values) == 0:
        return values
    out = np.zeros_like(values)
    out[0] = values[0]
    for i in range(1, len(values)):
        out[i] = decay * out[i - 1] + (1.0 - decay) * values[i]
    return out

def plot_learning_curve(
    returns, title="Learning curve", ylabel="Episodic return",
    save_path=None, smooth=0.0,
):
    """Plot a single learning curve (one list/array of per-epoch values)."""
    plot_learning_curves(
        {"run": returns}, title=title, ylabel=ylabel,
        save_path=save_path, smooth=smooth,
    )

def plot_learning_curves(
    curves, title="Learning curves", ylabel="Episodic return",
    save_path=None, smooth=0.0,
):
    """Plot multiple labelled curves on the same axes.

    Args:
        curves: dict mapping label -> iterable of per-epoch values.
        smooth: EMA decay in [0, 1). 0 = raw, ~0.9 = heavily smoothed.
    """
    plt.figure()
    for label, values in curves.items():
        values = np.asarray(values, dtype=np.float64)
        if smooth > 0:
            line, = plt.plot(values, alpha=0.25)
            plt.plot(_ema(values, smooth), label=label, color=line.get_color())
        else:
            plt.plot(values, label=label)
    plt.xlabel("Training epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"saved {save_path}")
    plt.show()


def plot_loss_curves(losses, title="Loss curves", save_path=None, smooth=0.0):
    """Plot one or more loss curves. `losses` is a dict label -> iterable."""
    plot_learning_curves(
        losses, title=title, ylabel="Loss",
        save_path=save_path, smooth=smooth,
    )

# ---------------------------------------------------------------------------
# Partial observability: extract only angular velocity from state
# ---------------------------------------------------------------------------

def state_extract(state: np.ndarray) -> np.ndarray:
    """Keep only angular velocity (index 2) from the full state."""
    return np.array([state[2]], dtype=np.float32)

# ---------------------------------------------------------------------------
# Networks
# ---------------------------------------------------------------------------

class RecurrentActor(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, action_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.mu_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)  # replaces nn.Parameter

    def forward(self, x, hidden=None):
        out, hidden = self.lstm(x, hidden)
        out = torch.relu(self.fc(out))
        mu = self.mu_head(out)
        log_std = self.log_std_head(out).clamp(-3, -0.5)
        std = log_std.exp()
        return mu, std, hidden

    def init_hidden(self, batch_size: int = 1):
        h = torch.zeros(1, batch_size, self.hidden_dim)
        c = torch.zeros(1, batch_size, self.hidden_dim)
        return (h, c)

class RecurrentCritic(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc   = nn.Linear(hidden_dim, 1)

    def forward(self, x, hidden=None):
        """
        x      : (B, T, input_dim)
        returns: values (B, T, 1), hidden
        """
        out, hidden = self.lstm(x, hidden)
        return self.fc(out), hidden

    def init_hidden(self, batch_size: int = 1):
        h = torch.zeros(1, batch_size, self.hidden_dim)
        c = torch.zeros(1, batch_size, self.hidden_dim)
        return (h, c)

# ---------------------------------------------------------------------------
# Replay buffer
# ---------------------------------------------------------------------------

class Buffer:
    def __init__(self, sdim, adim, size, sdtype=np.float32, adtype=np.float32):
        self.states = np.zeros((size, sdim), dtype=sdtype)
        self.actions = np.zeros((size, adim), dtype=adtype)
        self.rewards = np.zeros((size, 1), dtype=np.float32)
        self.ret_to_go = np.zeros((size, 1), dtype=np.float32)
        self.dones = np.zeros((size, 1), dtype=bool)
        self.i = 0
        self.size = size
        self.max_i = 0

    def add(self, state, action, reward, done):
        self.states[self.i] = state
        self.actions[self.i] = action
        self.rewards[self.i] = reward
        self.dones[self.i] = done
        self.i = (self.i + 1) % self.size
        self.max_i = min(self.max_i + 1, self.size)

    def calc_reward_to_go(self, gamma: float = 0.975):
        ret = 0.0
        for i in reversed(range(self.max_i)):
            if self.dones[i, 0]:
                ret = 0.0
            ret = self.rewards[i, 0] + gamma * ret
            self.ret_to_go[i, 0] = ret

    def sample(self, batch_size: int):
        """Sample individual transitions (used for critic regression)."""
        upper = max(self.max_i - 1, 1)
        idxs  = np.random.randint(0, upper, size=batch_size)
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

    def sample_sequence(self, batch_size: int, seq_len: int):
        """Sample contiguous, episode-safe sequences for recurrent training."""
        sequences = []
        for _ in range(batch_size):
            for _ in range(10_000): # retry until valid
                start = np.random.randint(0, self.max_i - seq_len)
                if not self.dones[start : start + seq_len - 1].any():
                    break
            end = start + seq_len
            sequences.append((
                self.states[start:end],
                self.actions[start:end],
                self.rewards[start:end],
                self.ret_to_go[start:end],
                self.dones[start:end],
            ))
        states, actions, rewards, rtg, dones = zip(*sequences)
        # each array: (batch_size, seq_len, dim)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(rtg),
            np.array(dones),
        )

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def rescale_actions(action: np.ndarray, amin: float, amax: float) -> np.ndarray:
    """Map tanh output (-1,1) → env range [amin, amax]."""
    return amin + (action + 1.0) * 0.5 * (amax - amin)

def act(policy: RecurrentActor, state_seq: np.ndarray, hidden):
    """
    state_seq : (T, state_dim) — recent context window
    Returns action (action_dim,) and updated hidden state.
    """
    x = torch.tensor(state_seq, dtype=torch.float32).unsqueeze(0) # (1, T, D)
    with torch.no_grad():
        mu, std, hidden = policy(x, hidden)
    mu  = mu[:, -1, :] # last timestep only (1, A)
    std = std[:, -1, :]
    dist = Normal(mu, std)
    action = dist.sample()
    return action.squeeze(0).numpy(), hidden

def collect_data(num_steps: int, env, agent: RecurrentActor, seq_len: int = 4):
    """
    Roll out `agent` for `num_steps` steps in `env`.

    Returns:
        buffer           — populated Buffer
        avg_step_reward  — mean per-step reward
        avg_ep_reward    — mean per-episode return
    """
    s, _ = env.reset()
    s = state_extract(s)
    sdim = len(s)
    adim = env.action_space.shape[0]

    buffer = Buffer(sdim=sdim, adim=adim, size=num_steps)
    hidden = agent.init_hidden(batch_size=1)

    state_seq = deque([s] * seq_len, maxlen=seq_len)

    episode_rewards: list[float] = []
    ep_reward = 0.0

    for _ in range(num_steps):
        seq_array = np.array(state_seq) # (T, D)
        a, hidden = act(policy=agent, state_seq=seq_array, hidden=hidden)
        a_scaled = rescale_actions(a, env.action_space.low[0], env.action_space.high[0])
        s2, r, term, trunc, _ = env.step(a_scaled)
        s2 = state_extract(s2)
        done = term or trunc

        buffer.add(state=s, action=a_scaled, reward=r, done=done)
        ep_reward += r
        state_seq.append(s2)
        s = s2

        if done:
            episode_rewards.append(ep_reward)
            ep_reward = 0.0
            s, _ = env.reset()
            s    = state_extract(s)
            hidden = agent.init_hidden(batch_size=1)
            state_seq = deque([s] * seq_len, maxlen=seq_len)

    if ep_reward != 0.0:
        episode_rewards.append(ep_reward)

    avg_step_reward = float(np.mean(buffer.rewards[: buffer.max_i]))
    avg_ep_reward   = float(np.mean(episode_rewards)) if episode_rewards else 0.0
    return buffer, avg_step_reward, avg_ep_reward

# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def _log_prob(policy, states, actions, hidden=None):
    mu, std, _ = policy(states, hidden)
    mu  = mu[:, -1, :]
    std = std[:, -1, :]
    dist = Normal(mu, std)
    return dist.log_prob(actions).sum(dim=-1).clamp(-10, 2)

def reinforce_signal(policy, states, actions, rewards_to_go, avg_rwd=0.0, use_avg=False):
    log_probs = _log_prob(policy, states, actions)
    advantage = rewards_to_go.squeeze(-1)
    if use_avg:
        advantage = advantage - avg_rwd
    advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
    return -(advantage * log_probs).mean()

def reinforce_rwd_signal(policy, states, actions, rewards):
    log_probs = _log_prob(policy, states, actions)
    return -(rewards.squeeze(-1) * log_probs).mean()

def reinforce_adv_signal(policy, states, actions, advantages):
    """Policy-gradient loss: -E[A_t * log π(a|s)]."""
    log_probs = _log_prob(policy, states, actions)
    return -(advantages.squeeze(-1) * log_probs).mean()

def ppo_surrogate_loss(policy, states, actions, advantages, old_log_probs, eps_clip=0.2):
    """Clipped PPO surrogate (actor only)."""
    mu, std, _ = policy(states, hidden=None)
    mu = mu[:, -1, :]
    std = std[:, -1, :]
    dist = Normal(mu, std)
    log_probs = dist.log_prob(actions).sum(-1)
    ratios = torch.exp(log_probs - old_log_probs)
    clipped = torch.clamp(ratios, 1 - eps_clip, 1 + eps_clip)
    adv = advantages.squeeze(-1)
    return -torch.min(ratios * adv, clipped * adv).mean()


def ppo_total_loss(policy, critic, states, actions, advantages, returns, old_log_probs, eps_clip=0.2, c1=0.5, c2=0.01):
    """Combined PPO loss: actor + value + entropy."""
    mu, std, _ = policy(states, hidden=None)
    mu  = mu[:, -1, :]
    std = std[:, -1, :]
    dist = Normal(mu, std)
    log_probs = dist.log_prob(actions).sum(-1)

    ratios = torch.exp(log_probs - old_log_probs)
    adv = advantages.squeeze(-1)
    surr = -torch.min(
            ratios * adv,
            torch.clamp(ratios, 1 - eps_clip, 1 + eps_clip) * adv
        ).mean()

    values, _ = critic(states, hidden=None)          # (B, T, 1)
    val_loss = mse_loss(values[:, -1, :].squeeze(-1), returns.squeeze(-1))

    entropy = dist.entropy().sum(-1).mean()

    return surr + c1 * val_loss - c2 * entropy
# ---------------------------------------------------------------------------
# GAE
# ---------------------------------------------------------------------------
 
def compute_gae(rewards, values, next_values, dones, gamma=0.975, lam=0.95):
    """
    All inputs: (N, 1) numpy arrays.
    Returns  : (N, 1) numpy array of advantages.
    """
    advantages = np.zeros_like(rewards, dtype=np.float32)
    last_adv = 0.0
    dones_f = dones.astype(np.float32)
    N = len(rewards)
    for t in reversed(range(N)):
        mask = 1.0 - dones_f[t, 0]
        delta = rewards[t, 0] + gamma * next_values[t, 0] * mask - values[t, 0]
        last_adv = delta + gamma * lam * mask * last_adv
        advantages[t, 0] = last_adv
    return advantages
 
 
# ---------------------------------------------------------------------------
# Training routines
# ---------------------------------------------------------------------------
 
def train_vpg(
    epochs=300,
    episodes=10,
    updates=10,
    learning_rate=1e-3,
    hidden_size=64,
    batch_size=256,
    seq_len=4,
    use_avg=False,
    use_rwds=False,
    gamma=0.975,
):
    """Vanilla policy gradient (reinforce) with recurrent actor."""
    env = gym.make("Pendulum-v1")
    state_dim  = 1
    action_dim = env.action_space.shape[0]
    ep_len = env.spec.max_episode_steps
 
    policy = RecurrentActor(state_dim, hidden_size, action_dim)
    optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)
 
    returns_per_epoch = []
    losses_per_epoch  = []
    for epoch in range(epochs):
        with torch.no_grad():
            buffer, avg_step_rwd, avg_ep_rwd = collect_data(
                episodes * ep_len, env, policy, seq_len=seq_len
            )
        buffer.calc_reward_to_go(gamma)
 
        epoch_loss = 0.0
        for _ in range(updates):
            states, actions, rewards, rtg, _ = buffer.sample_sequence(batch_size, seq_len)
            s_t   = torch.as_tensor(states,             dtype=torch.float32)  # (B, T, 1)
            a_t   = torch.as_tensor(actions[:, -1, :],  dtype=torch.float32)  # (B, A) last step
            rtg_t = torch.as_tensor(rtg[:, -1, :],      dtype=torch.float32)  # (B, 1) last step
            r_t   = torch.as_tensor(rewards[:, -1, :],  dtype=torch.float32)
 
            optimizer.zero_grad()
            if use_rwds:
                loss = reinforce_rwd_signal(policy, s_t, a_t, r_t)
            else:
                loss = reinforce_signal(policy, s_t, a_t, rtg_t, avg_step_rwd, use_avg)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=0.5)
            optimizer.step()
            epoch_loss += loss.item()
 
        avg_loss = epoch_loss / updates
        returns_per_epoch.append(avg_ep_rwd)
        losses_per_epoch.append(avg_loss)
        print(f"VPG epoch {epoch+1}/{epochs}: return={avg_ep_rwd:.2f}  loss={avg_loss:.4f}")
 
    return policy, returns_per_epoch, losses_per_epoch
 
 
def train_advantage_vpg(
    epochs=300,
    episodes=10,
    updates=10,
    critic_updates=80,
    learning_rate=1e-4,
    critic_lr=3e-4,
    hidden_size=64,
    batch_size=256,
    seq_len=4,
    gamma=0.975,
    lam=0.95,
):
    """Policy gradient with a recurrent V(s) baseline and GAE."""
    env = gym.make("Pendulum-v1")
    state_dim = 1
    action_dim = env.action_space.shape[0]
    ep_len = env.spec.max_episode_steps
 
    policy = RecurrentActor(state_dim, hidden_size, action_dim)
    critic = RecurrentCritic(state_dim, hidden_size)
    optimizer = torch.optim.Adam(policy.parameters(),  lr=learning_rate)
    cr_optimizer = torch.optim.Adam(critic.parameters(), lr=critic_lr)
 
    returns_per_epoch = []
    actor_losses_per_epoch = []
    critic_losses_per_epoch = []
    for epoch in range(epochs):
        with torch.no_grad():
            buffer, _, avg_ep_rwd = collect_data(
                episodes * ep_len, env, policy, seq_len=seq_len
            )
        buffer.calc_reward_to_go(gamma)
 
        # --- train critic: regress V(s) toward reward-to-go ---
        critic_loss_sum = 0.0
        for _ in range(critic_updates):
            states, _, _, rtg, _ = buffer.sample_sequence(batch_size, seq_len)
            s_t   = torch.as_tensor(states, dtype=torch.float32)  # (B, T, 1)
            rtg_t = torch.as_tensor(rtg[:, -1:, :], dtype=torch.float32)  # (B, 1, 1)
            cr_optimizer.zero_grad()
            values, _ = critic(s_t) # (B, T, 1)
            loss = mse_loss(values[:, -1:, :], rtg_t)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=0.5)
            cr_optimizer.step()
            critic_loss_sum += loss.item()
 
        # --- compute GAE over the full buffer ---
        all_s = torch.as_tensor(buffer.states[: buffer.max_i], dtype=torch.float32)
        all_s = all_s.unsqueeze(1) # (N, 1, 1) — T=1 per step
        with torch.no_grad():
            values, _ = critic(all_s)
        values = values.squeeze().numpy().reshape(-1, 1)
 
        next_values = np.zeros_like(values)
        next_values[:-1] = values[1:]
        next_values[buffer.dones[: buffer.max_i, 0]] = 0.0
 
        advantages = compute_gae(
            buffer.rewards[: buffer.max_i],
            values,
            next_values,
            buffer.dones[: buffer.max_i],
            gamma, lam,
        )
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
 
        # --- train actor with GAE advantages ---
        actor_loss_sum = 0.0
        for _ in range(updates):
            states, actions, _, _, _ = buffer.sample_sequence(batch_size, seq_len)
            idxs  = np.random.randint(0, buffer.max_i - seq_len, size=batch_size)
            adv_s = np.array([advantages[i + seq_len - 1] for i in idxs])  # (B, 1)
 
            s_t   = torch.as_tensor(states, dtype=torch.float32)
            a_t   = torch.as_tensor(actions[:, -1, :], dtype=torch.float32)
            adv_t = torch.as_tensor(adv_s, dtype=torch.float32)
 
            optimizer.zero_grad()
            loss = reinforce_adv_signal(policy, s_t, a_t, adv_t)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=0.5)
            optimizer.step()
            actor_loss_sum += loss.item()
 
        returns_per_epoch.append(avg_ep_rwd)
        actor_losses_per_epoch.append(actor_loss_sum / updates)
        critic_losses_per_epoch.append(critic_loss_sum / critic_updates)
        print(f"GAE epoch {epoch+1}/{epochs}: return={avg_ep_rwd:.2f}"
              f"  actor_loss={actor_losses_per_epoch[-1]:.4f}"
              f"  critic_loss={critic_losses_per_epoch[-1]:.4f}")
 
    return policy, returns_per_epoch, actor_losses_per_epoch, critic_losses_per_epoch
 
 
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
    """Full recurrent PPO with clipped surrogate + entropy bonus."""
    env        = gym.make("Pendulum-v1")
    state_dim  = 1
    action_dim = env.action_space.shape[0]
    ep_len     = env.spec.max_episode_steps
 
    policy       = RecurrentActor(state_dim, hidden_size, action_dim)
    critic       = RecurrentCritic(state_dim, hidden_size)
    optimizer    = torch.optim.Adam(policy.parameters(), lr=learning_rate)
    cr_optimizer = torch.optim.Adam(critic.parameters(), lr=learning_rate)
 
    returns_per_iter: list[float] = []
    losses_per_iter:  list[float] = []
 
    for k in range(iterations):
        # --- collect rollout ---
        buffer, _, _ = collect_data(steps_per_iter, env, policy, seq_len=1)
        buffer.calc_reward_to_go(gamma)
 
        N       = buffer.max_i
        states  = torch.tensor(buffer.states[:N],    dtype=torch.float32)   # (N, D)
        actions = torch.tensor(buffer.actions[:N],   dtype=torch.float32)   # (N, A)
        dones   = buffer.dones[:N]
        returns = torch.tensor(buffer.ret_to_go[:N], dtype=torch.float32)   # (N, 1)
 
        # --- GAE ---
        with torch.no_grad():
            values, _ = critic(states.unsqueeze(1))   # (N, 1, 1)
        values = values.squeeze().numpy().reshape(-1, 1)
 
        next_values      = np.zeros_like(values)
        next_values[:-1] = values[1:]
        next_values[dones[:, 0]] = 0.0
 
        advantages = compute_gae(
            buffer.rewards[:N], values, next_values, dones, gamma, lam
        )
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        advantages = torch.tensor(advantages, dtype=torch.float32)
 
        # --- compute old log-probs once ---
        old_policy = deepcopy(policy).eval()
        with torch.no_grad():
            mu, std, _ = old_policy(states.unsqueeze(1))
            old_lp     = Normal(mu[:, -1, :], std[:, -1, :]).log_prob(actions).sum(-1)
 
        # --- PPO update ---
        total_loss = 0.0
        for _ in range(sgd_epochs):
            idxs = np.random.randint(0, N, size=minibatch_size)
 
            b_s   = states[idxs].unsqueeze(1)    # (B, 1, D)
            b_a   = actions[idxs]                 # (B, A)
            b_adv = advantages[idxs]              # (B, 1)
            b_ret = returns[idxs]                 # (B, 1)
            b_olp = old_lp[idxs]                  # (B,)
 
            mu, std, _ = policy(b_s)
            dist       = Normal(mu[:, -1, :], std[:, -1, :])
            log_probs  = dist.log_prob(b_a).sum(-1)
            ratios     = torch.exp(log_probs - b_olp)
            adv        = b_adv.squeeze(-1)
 
            surr1      = ratios * adv
            surr2      = torch.clamp(ratios, 1 - eps_clip, 1 + eps_clip) * adv
            actor_loss = -torch.min(surr1, surr2).mean() if clip else -(ratios * adv).mean()
 
            val_pred, _ = critic(b_s)
            val_loss    = mse_loss(val_pred[:, -1, :].squeeze(-1), b_ret.squeeze(-1))
 
            entropy     = dist.entropy().sum(-1).mean()
            loss        = actor_loss + c1 * val_loss - c2 * entropy
 
            optimizer.zero_grad()
            cr_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=0.5)
            torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=0.5)
            optimizer.step()
            cr_optimizer.step()
            total_loss += loss.item()
 
        avg_loss = total_loss / sgd_epochs
 
        # track episode returns from this iteration's buffer
        ep_returns = []
        ep_r       = 0.0
        for i in range(N):
            ep_r += buffer.rewards[i, 0]
            if buffer.dones[i, 0]:
                ep_returns.append(ep_r)
                ep_r = 0.0
        if ep_r != 0.0:
            ep_returns.append(ep_r)
        avg_return = float(np.mean(ep_returns)) if ep_returns else 0.0
 
        returns_per_iter.append(avg_return)
        losses_per_iter.append(avg_loss)
        print(f"PPO iter {k+1}/{iterations}: return={avg_return:.2f}  loss={avg_loss:.4f}")
 
    return policy, returns_per_iter, losses_per_iter
 
 
# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
 
if __name__ == "__main__":
    # --- Task 1: sanity-check random policy ---
    env    = gym.make("Pendulum-v1")
    policy = RecurrentActor(input_dim=1, hidden_dim=64, action_dim=1)
    buf, step_r, ep_r = collect_data(1000, env, policy)
    print(f"Random policy | avg step reward: {step_r:.3f} | avg ep reward: {ep_r:.2f}")
 
    # --- Task 2: VPG with two learning rates ---
    p1, ret1, loss1 = train_vpg(epochs=200, learning_rate=1e-3)
    p2, ret2, loss2 = train_vpg(epochs=200, learning_rate=2e-4)
    plot_learning_curves(
        {"lr=1e-3": ret1, "lr=2e-4": ret2},
        title="Task 2: VPG learning rates", smooth=0.9,
    )
    plot_loss_curves(
        {"lr=1e-3": loss1, "lr=2e-4": loss2},
        title="Task 2: VPG loss curves", smooth=0.9,
    )
 
    # --- Task 3: reward-to-go vs GAE ---
    _, ret_rtg, loss_rtg = train_vpg(epochs=200, learning_rate=3e-4)
    _, ret_gae, actor_losses_gae, critic_losses_gae = train_advantage_vpg(epochs=200, learning_rate=3e-4)
    plot_learning_curves(
        {"rewards-to-go": ret_rtg, "GAE": ret_gae},
        title="Task 3: RTG vs GAE", smooth=0.9,
    )
    plot_loss_curves(
        {"RTG actor": loss_rtg},
        title="Task 3: VPG loss", smooth=0.9,
    )
    plot_loss_curves(
        {"GAE actor": actor_losses_gae, "GAE critic": critic_losses_gae},
        title="Task 3: GAE loss curves", smooth=0.9,
    )
 
    # --- Task 4 & 5: PPO clipped vs unclipped ---
    _, ret_clip,   loss_clip   = train_ppo(iterations=200, clip=True)
    _, ret_noclip, loss_noclip = train_ppo(iterations=200, clip=False)
    plot_learning_curves(
        {"clipped": ret_clip, "unclipped": ret_noclip},
        title="Task 4: PPO clipped vs unclipped", smooth=0.9,
    )
    plot_loss_curves(
        {"clipped": loss_clip, "unclipped": loss_noclip},
        title="Task 4: PPO loss curves", smooth=0.9,
    )
 
    _, ret_ppo, loss_ppo = train_ppo(iterations=200, clip=True)
    plot_learning_curves({"PPO": ret_ppo}, title="Task 5: Full PPO", smooth=0.9)
    plot_loss_curves({"PPO": loss_ppo},    title="Task 5: PPO loss", smooth=0.9)
    