"""
HW4 — Task 1: Replay buffer and environment interaction.

Complete the four TODO items below before moving on to vpg.py.
"""

import numpy as np
import torch as th
import torch.nn as nn
from torch.distributions import Normal
import gymnasium as gym

from Modules import NormalModule


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
        ret = 0.0
        for i in reversed(range(self.max_i)):
            if self.dones[i, 0]:
                ret = 0.0
            ret = self.rewards[i, 0] + gamma * ret
            self.ret_to_go[i, 0] = ret


def collect_data(size, env, agent, title="collecting"):
    """Roll out `agent` (a policy network) in `env` for `size` steps.

    Returns:
        buffer  — a populated Buffer
        avg_rwd — average per-step reward observed during the rollout
    """
    s, _ = env.reset()
    a = act(policy=agent, state=s)
    buffer = Buffer(sdim=len(s), adim=a.shape[0], size=size)
    cumulative_reward = 0.0
    episode_rewards = []
    episode_reward = 0.0
    for _ in range(size):
        a = act(policy=agent, state=s)
        a_scaled = rescale_actions(a, env.action_space.low[0], env.action_space.high[0])
        s2, r, terminated, truncated, _ = env.step(a_scaled)
        done = terminated or truncated
        buffer.add(state=s, action=a, reward=r, done=done)
        cumulative_reward += r
        episode_reward += r
        if done:
            episode_rewards.append(episode_reward)
            episode_reward = 0.0
            s, _ = env.reset()
        else:
            s = s2
    buffer.calc_reward_to_go()
    avg_step_reward = cumulative_reward / size
    if episode_rewards:
        avg_episode_reward = np.mean(episode_rewards)
    else:
        avg_episode_reward = 0.0
    return buffer, avg_step_reward, avg_episode_reward


def act(policy, state):
    """Sample a continuous action a ~ N(mu(state), sigma) from the policy."""
    state_tensor = th.tensor(state, dtype=th.float32)
    mu, log_std = policy(state_tensor)
    std = th.exp(log_std)
    distribution = Normal(mu, std)
    action = distribution.sample()
    return action.detach().numpy().reshape(-1)
    


def rescale_actions(action, amin, amax):
    """Rescale a tanh-squashed action from (-1, 1) to the env range [amin, amax]."""
    return amin + (action + 1.0) * 0.5 * (amax-amin)

