"""
HW4 — Tasks 4 & 5: PPO surrogate objective and full PPO algorithm.

Depends on: buffer.py (Task 1), vpg.py (Task 2), gae.py (Task 3).
"""

import numpy as np
import torch as th
from torch.nn.functional import mse_loss
from torch.distributions import Normal
import gymnasium as gym

from buffer import Buffer, collect_data, act, rescale_actions
from vpg import _log_prob, build_actor
from gae import build_critic, compute_gae


# ---------------------------------------------------------------------------
# Internal helper (provided — do not modify)
# ---------------------------------------------------------------------------

def _critic_values(critic, buffer):
    """Run the critic on every stored state, returning (values, next_values)."""
    states = th.as_tensor(buffer.states[: buffer.max_i], dtype=th.float32)
    with th.no_grad():
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
    log_probs = _log_prob(policy, states, actions)
    ratios = th.exp(log_probs - old_log_probs)
    if clip:
        clipped_ratios = th.clamp(ratios, 1.0 - eps_clip, 1.0 + eps_clip)
        unclipped = ratios * advantages
        clipped = clipped_ratios * advantages
        loss = -th.min(unclipped, clipped).mean()
    else:
        loss = -(ratios * advantages).mean()
    return loss


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
    surr_loss = ppo_surrogate_loss(
        policy,
        states,
        actions,
        advantages,
        old_log_probs,
        eps_clip=eps_clip,
        clip=clip,
    )
    values = critic(states)
    value_loss = mse_loss(values, returns)
    mu, sigma = policy(states)
    entropy = Normal(mu, sigma).entropy().sum(dim=-1).mean()
    return surr_loss + c1 * value_loss - c2 * entropy


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
    optimizer    = th.optim.Adam(policy.parameters(), lr=learning_rate)
    cr_optimizer = th.optim.Adam(critic.parameters(), lr=learning_rate)

    returns_per_iter = []
    losses_per_iter  = []

    for k in range(iterations):
        buffer = Buffer(sdim=state_dim, adim=action_dim, size=steps_per_iter)
        s, _ = env.reset()
        episode_rewards = []
        episode_reward = 0.0

        with th.no_grad():
            for _ in range(steps_per_iter):
                a = act(policy, s)
                a_scaled = rescale_actions(a, env.action_space.low[0], env.action_space.high[0])
                s2, r, terminated, truncated, _ = env.step(a_scaled)
                done = terminated or truncated
                buffer.add(state=s, action=a, reward=r, done=done)
                episode_reward += r
                if done:
                    episode_rewards.append(episode_reward)
                    episode_reward = 0.0
                    s, _ = env.reset()
                else:
                    s = s2

        if episode_reward != 0.0:
            episode_rewards.append(episode_reward)

        buffer.calc_reward_to_go(gamma)

        all_states = th.as_tensor(buffer.states[: buffer.max_i], dtype=th.float32)
        all_actions = th.as_tensor(buffer.actions[: buffer.max_i], dtype=th.float32)
        with th.no_grad():
            values = critic(all_states).numpy()
        next_values = np.zeros_like(values)
        next_values[:-1] = values[1:]
        next_values[buffer.dones[: buffer.max_i, 0]] = 0.0

        advantages = compute_gae(
            rewards=buffer.rewards[: buffer.max_i],
            values=values,
            next_values=next_values,
            dones=buffer.dones[: buffer.max_i],
            gamma=gamma,
            lam=lam,
        )
        returns = advantages + values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        all_returns = th.as_tensor(returns, dtype=th.float32)
        all_advantages = th.as_tensor(advantages, dtype=th.float32)
        old_log_probs = _log_prob(policy, all_states, all_actions).detach()

        iter_loss = 0.0
        batch_count = 0
        for _ in range(sgd_epochs):
            idxs = np.random.randint(0, buffer.max_i, size=minibatch_size)
            batch_states = all_states[idxs]
            batch_actions = all_actions[idxs]
            batch_advantages = all_advantages[idxs]
            batch_returns = all_returns[idxs]
            batch_old_log_probs = old_log_probs[idxs]

            optimizer.zero_grad()
            cr_optimizer.zero_grad()
            loss = ppo_total_loss(
                policy,
                critic,
                batch_states,
                batch_actions,
                batch_advantages,
                batch_returns,
                batch_old_log_probs,
                eps_clip=eps_clip,
                c1=c1,
                c2=c2,
                clip=clip,
            )
            loss.backward()
            optimizer.step()
            cr_optimizer.step()

            iter_loss += loss.item()
            batch_count += 1

        avg_loss = iter_loss / max(batch_count, 1)
        avg_return = float(np.mean(episode_rewards)) if len(episode_rewards) > 0 else 0.0
        returns_per_iter.append(avg_return)
        losses_per_iter.append(avg_loss)
        print(f"ppo iter {k + 1}/{iterations}: return={avg_return:.2f} loss={avg_loss:.4f}")

    return policy, returns_per_iter, losses_per_iter


if __name__ == "__main__":
    from plotting import plot_learning_curves, plot_loss_curves
    from video import record_video, generate_strobe

    # --- Task 4: clipped vs unclipped ---
    _, ret_clip,   loss_clip   = train_ppo(iterations=50, clip=True)
    _, ret_noclip, loss_noclip = train_ppo(iterations=50, clip=False)
    plot_learning_curves(
        {"clipped": ret_clip, "unclipped": ret_noclip},
        title="Task 4: PPO clipped vs unclipped",
    )
    plot_loss_curves(
        {"clipped": loss_clip, "unclipped": loss_noclip},
        title="Task 4: PPO loss curves",
    )

    # --- Task 5: full PPO ---
    policy, ret_ppo, loss_ppo = train_ppo(iterations=500)
    plot_learning_curves({"PPO": ret_ppo}, title="Task 5: Full PPO")
    plot_loss_curves({"PPO": loss_ppo}, title="Task 5: Total loss")
    record_video(policy, path="videos/task5_ppo.mp4")            # optional
    generate_strobe(policy, path="videos/task5_ppo_strobe.png")  # optional
