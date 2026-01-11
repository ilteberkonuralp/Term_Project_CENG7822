# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import random

class DiscreteHERPolicyNetwork(nn.Module):
    def __init__(self, num_states, num_actions, hidden_size=128):
        super(DiscreteHERPolicyNetwork, self).__init__()
        self.embedding = nn.Embedding(num_states, hidden_size)
        self.linear1 = nn.Linear(hidden_size * 2, hidden_size)
        self.linear2 = nn.Linear(hidden_size, num_actions)

    def forward(self, state_idx, goal_idx):
        s_emb = self.embedding(state_idx)
        g_emb = self.embedding(goal_idx)
        x = torch.cat([s_emb, g_emb], dim=-1)
        x = F.relu(self.linear1(x))
        logits = self.linear2(x)
        probs = F.softmax(logits, dim=-1)
        z = (probs == 0.0).float() * 1e-8
        log_probs = torch.log(probs + z)
        return probs, log_probs

    def sample(self, state_idx, goal_idx):
        probs, log_probs = self.forward(state_idx, goal_idx)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item(), probs, log_probs

class DiscreteHERQNetwork(nn.Module):
    def __init__(self, num_states, num_actions, hidden_size=128):
        super(DiscreteHERQNetwork, self).__init__()
        self.embedding = nn.Embedding(num_states, hidden_size)
        self.linear1 = nn.Linear(hidden_size * 2, hidden_size)
        self.linear2 = nn.Linear(hidden_size, num_actions)

    def forward(self, state_idx, goal_idx):
        s_emb = self.embedding(state_idx)
        g_emb = self.embedding(goal_idx)
        x = torch.cat([s_emb, g_emb], dim=-1)
        x = F.relu(self.linear1(x))
        q_values = self.linear2(x)
        return q_values

class SAC_HER_Agent():
    def __init__(self, info):
        self.num_actions = info["num_actions"]
        self.num_states = info["num_states"]
        self.grid_width = info.get("grid_width", 10) 
        self.gamma = info.get("discount", 0.99)
        self.tau = 0.005
        self.lr = info.get("step_size", 0.001)
        self.batch_size = 64
        self.memory_capacity = 50000 
        self.device = torch.device("cpu")
        self.k_future = 4 

        # --- Automatic Entropy Tuning ---
        self.target_entropy = -0.98 * np.log(1 / self.num_actions) 
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optim = optim.Adam([self.log_alpha], lr=self.lr)
        self.alpha = self.log_alpha.exp()

        self.critic1 = DiscreteHERQNetwork(self.num_states, self.num_actions).to(self.device)
        self.critic2 = DiscreteHERQNetwork(self.num_states, self.num_actions).to(self.device)
        self.critic1_target = DiscreteHERQNetwork(self.num_states, self.num_actions).to(self.device)
        self.critic2_target = DiscreteHERQNetwork(self.num_states, self.num_actions).to(self.device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        self.actor = DiscreteHERPolicyNetwork(self.num_states, self.num_actions).to(self.device)

        self.q_optim = optim.Adam(list(self.critic1.parameters()) + list(self.critic2.parameters()), lr=self.lr)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=self.lr)

        self.replay_buffer = [] 
        self.episode_buffer = [] 
        self.rand_generator = np.random.RandomState(info["seed"])

    def save_model(self, filename="sac_her_checkpoint"):
        torch.save({
            'actor': self.actor.state_dict(),
            'critic1': self.critic1.state_dict(),
            'critic2': self.critic2.state_dict(),
            'log_alpha': self.log_alpha
        }, filename + ".pth")

    def load_model(self, filename="sac_her_checkpoint"):
        if not os.path.exists(filename + ".pth"): return
        checkpoint = torch.load(filename + ".pth", map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic1.load_state_dict(checkpoint['critic1'])
        self.critic2.load_state_dict(checkpoint['critic2'])
        self.log_alpha = checkpoint['log_alpha']
        self.alpha = self.log_alpha.exp()
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())

    def _get_idx(self, state):
        if hasattr(state, '__len__'): return int(self.grid_width * state[0] + state[1])
        return int(state)

    def compute_reward(self, state_idx, goal_idx):
        return 0.0 if int(state_idx) == int(goal_idx) else -1.0

    def agent_start(self, state, goal):
        self.curr_state_idx = self._get_idx(state)
        self.curr_goal_idx = self._get_idx(goal)
        self.episode_buffer = []
        s_tensor = torch.LongTensor([self.curr_state_idx]).to(self.device)
        g_tensor = torch.LongTensor([self.curr_goal_idx]).to(self.device)
        with torch.no_grad():
            action, _, _ = self.actor.sample(s_tensor, g_tensor)
        self.prev_action = action
        return action

    def agent_step(self, reward, next_state, done):
        next_state_idx = self._get_idx(next_state)
        self.episode_buffer.append((self.curr_state_idx, self.prev_action, reward, next_state_idx, done, self.curr_goal_idx))
        self.curr_state_idx = next_state_idx
        if not done:
            s_tensor = torch.LongTensor([self.curr_state_idx]).to(self.device)
            g_tensor = torch.LongTensor([self.curr_goal_idx]).to(self.device)
            with torch.no_grad():
                action, _, _ = self.actor.sample(s_tensor, g_tensor)
            self.prev_action = action
            return action
        else:
            return None

    def agent_end(self):
        for t, transition in enumerate(self.episode_buffer):
            state, action, env_reward, next_state, done, original_goal = transition
            real_reward = self.compute_reward(next_state, original_goal)
            self.replay_buffer.append((state, action, real_reward, next_state, done, original_goal))
            
            future_indices = range(t + 1, len(self.episode_buffer))
            if len(future_indices) > 0:
                count = min(self.k_future, len(future_indices))
                selected_indices = random.sample(future_indices, count)
                for idx in selected_indices:
                    achieved_goal = self.episode_buffer[idx][3]
                    new_reward = self.compute_reward(next_state, achieved_goal)
                    new_done = (next_state == achieved_goal)
                    self.replay_buffer.append((state, action, new_reward, next_state, new_done, achieved_goal))

        if len(self.replay_buffer) > self.memory_capacity:
            self.replay_buffer = self.replay_buffer[-self.memory_capacity:]

        q_loss_sum, a_loss_sum = 0, 0
        updates = len(self.episode_buffer)
        for _ in range(updates):
             q, a = self.update_parameters()
             q_loss_sum += q
             a_loss_sum += a
        return q_loss_sum / max(1, updates), a_loss_sum / max(1, updates)

    def update_parameters(self):
        if len(self.replay_buffer) < self.batch_size: return 0, 0
        indices = self.rand_generator.randint(0, len(self.replay_buffer), size=self.batch_size)
        batch = [self.replay_buffer[i] for i in indices]
        state_b, action_b, reward_b, next_state_b, done_b, goal_b = zip(*batch)

        state_batch = torch.LongTensor(state_b).to(self.device)
        action_batch = torch.LongTensor(action_b).unsqueeze(1).to(self.device)
        reward_batch = torch.FloatTensor(reward_b).unsqueeze(1).to(self.device)
        next_state_batch = torch.LongTensor(next_state_b).to(self.device)
        done_batch = torch.FloatTensor(done_b).unsqueeze(1).to(self.device)
        goal_batch = torch.LongTensor(goal_b).to(self.device)

        with torch.no_grad():
            next_probs, next_log_probs = self.actor(next_state_batch, goal_batch)
            q1_next = self.critic1_target(next_state_batch, goal_batch)
            q2_next = self.critic2_target(next_state_batch, goal_batch)
            min_q_next = torch.min(q1_next, q2_next)
            
            # --- FIX: EXACT EXPECTATION FOR DISCRETE TARGET ---
            # V(s') = Sum[ pi(a'|s') * (Q(s',a') - alpha * log_pi(a'|s')) ]
            target_v = (next_probs * (min_q_next - self.alpha * next_log_probs)).sum(dim=1, keepdim=True)
            target_q = reward_batch + (1 - done_batch) * self.gamma * target_v

        q1 = self.critic1(state_batch, goal_batch).gather(1, action_batch)
        q2 = self.critic2(state_batch, goal_batch).gather(1, action_batch)
        q_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

        self.q_optim.zero_grad()
        q_loss.backward()
        self.q_optim.step()

        probs, log_probs = self.actor(state_batch, goal_batch)
        with torch.no_grad():
            q1_val = self.critic1(state_batch, goal_batch)
            q2_val = self.critic2(state_batch, goal_batch)
            min_q = torch.min(q1_val, q2_val)
        
        actor_loss = (probs * (self.alpha * log_probs - min_q)).sum(dim=1).mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        # --- FIX: Alpha Loss ---
        alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()
        self.alpha = self.log_alpha.exp()

        for target, param in zip(self.critic1_target.parameters(), self.critic1.parameters()):
            target.data.copy_(target.data * (1.0 - self.tau) + param.data * self.tau)
        for target, param in zip(self.critic2_target.parameters(), self.critic2.parameters()):
            target.data.copy_(target.data * (1.0 - self.tau) + param.data * self.tau)
            
        return q_loss.item(), actor_loss.item()