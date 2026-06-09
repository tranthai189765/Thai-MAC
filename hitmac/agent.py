"""Rollout agent: collects trajectories and performs the A3C update."""

from __future__ import division

import torch
import numpy as np
from torch.autograd import Variable

from .utils import ensure_shared_grads


class Agent(object):
    """Owns a policy, steps the environment and optimizes with A3C + GAE."""

    def __init__(self, model, env, args, state, device):
        self.model = model
        self.env = env
        self.eps_len = 0
        self.eps_num = 0
        # Both the executor and coordinator currently use 4 cameras.
        self.num_agents = 4
        self.dim_action = 1
        self.args = args
        self.values = []
        self.log_probs = []
        self.rewards = []
        self.entropies = []
        self.rewards_eps = []
        self.done = True
        self.info = None
        self.reward = 0
        self.device = device
        self.lstm_out = args.lstm_out
        self.reward_mean = None
        self.reward_std = 1
        self.num_steps = 0
        self.n_steps = 0
        self.vk = 0
        self.state = state
        self.hxs = torch.zeros(self.num_agents, self.lstm_out).to(device)
        self.cxs = torch.zeros(self.num_agents, self.lstm_out).to(device)
        self.rank = 0
        self.rotation = 0

    def action_train(self):
        """Take one environment step and buffer the transition for training."""
        self.n_steps += 1
        value_multi, actions, entropy, log_prob = self.model(self.state)

        state_multi, reward_multi, self.done, self.info = self.env.step(actions)
        if isinstance(self.done, list):
            self.done = np.sum(self.done)
        self.state = torch.from_numpy(np.array(state_multi)).float().to(self.device)
        self.reward_org = reward_multi.copy()
        if self.args.norm_reward:
            reward_multi = self.reward_normalizer(reward_multi)
        self.reward = torch.tensor(reward_multi).float().to(self.device)
        self.eps_len += 1
        self.values.append(value_multi)
        self.entropies.append(entropy)
        self.log_probs.append(log_prob)
        self.rewards.append(self.reward.unsqueeze(1))

    def action_test(self):
        """Take one greedy environment step (no gradient, no buffering)."""
        with torch.no_grad():
            value_multi, actions, entropy, log_prob = self.model(self.state, True)

        state_multi, self.reward, self.done, self.info = self.env.step(actions)
        if isinstance(self.done, list):
            self.done = np.sum(self.done)
        self.state = torch.from_numpy(np.array(state_multi)).float().to(self.device)
        self.eps_len += 1

    def reset(self):
        """Reset the environment and the recurrent/exploration state."""
        obs = self.env.reset()
        self.state = torch.from_numpy(np.array(obs)).float().to(self.device)
        self.eps_len = 0
        self.eps_num += 1
        self.reset_rnn_hidden()
        self.model.sample_noise()

    def clean_buffer(self, done):
        """Clear the rollout buffers; also clear the episode buffer when done."""
        self.values = []
        self.log_probs = []
        self.entropies = []
        self.rewards = []
        if done:
            self.rewards_eps = []
        return self

    def reward_normalizer(self, reward):
        """Online (Welford) standardization of the reward signal."""
        reward = np.array(reward)
        self.num_steps += 1
        if self.num_steps == 1:
            self.reward_mean = reward
            self.vk = 0
            self.reward_std = 1
        else:
            delt = reward - self.reward_mean
            self.reward_mean = self.reward_mean + delt / self.num_steps
            self.vk = self.vk + delt * (reward - self.reward_mean)
            self.reward_std = np.sqrt(self.vk / (self.num_steps - 1))
        reward = (reward - self.reward_mean) / (self.reward_std + 1e-8)
        return reward

    def reset_rnn_hidden(self):
        self.cxs = Variable(torch.zeros(self.num_agents, self.lstm_out).to(self.device))
        self.hxs = Variable(torch.zeros(self.num_agents, self.lstm_out).to(self.device))

    def update_rnn_hidden(self):
        self.cxs = Variable(self.cxs.data)
        self.hxs = Variable(self.hxs.data)

    def optimize(self, params, optimizer, shared_model, training_mode, device_share):
        """Compute the A3C policy/value losses with GAE and apply the update."""
        R = torch.zeros(len(self.rewards[0]), 1).to(self.device)
        if not self.done:
            # Bootstrap from the value of the current (non-terminal) state.
            value_multi, *_ = self.model(self.state)
            for i in range(len(self.rewards[0])):  # per agent
                R[i][0] = value_multi[i].data

        self.values.append(Variable(R).to(self.device))

        batch_size = len(self.entropies[0][0])
        policy_loss = torch.zeros(batch_size, 1).to(self.device)
        value_loss = torch.zeros(1, 1).to(self.device)
        entropies = torch.zeros(batch_size, self.dim_action).to(self.device)
        w_entropies = float(self.args.entropy)

        R = Variable(R, requires_grad=True).to(self.device)
        gae = torch.zeros(1, 1).to(self.device)

        for i in reversed(range(len(self.rewards))):
            R = self.args.gamma * R + self.rewards[i]
            advantage = R - self.values[i]
            value_loss = value_loss + 0.5 * advantage.pow(2)
            # Generalized Advantage Estimation.
            delta_t = self.rewards[i] + self.args.gamma * self.values[i + 1].data - self.values[i].data
            gae = gae * self.args.gamma * self.args.tau + delta_t
            policy_loss = policy_loss \
                - (self.log_probs[i] * Variable(gae)) \
                - (w_entropies * self.entropies[i])
            entropies += self.entropies[i].sum()

        self.model.zero_grad()
        loss = policy_loss.sum() + 0.5 * value_loss.sum()
        loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(params, 50)
        ensure_shared_grads(self.model, shared_model, self.device, device_share)
        optimizer.step()

        self.clean_buffer(self.done)
        return policy_loss, value_loss, entropies
