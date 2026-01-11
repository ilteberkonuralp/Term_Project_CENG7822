# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import random

def quantile_huber_loss_f(quantiles, target_quantiles):
    pairwise_delta = target_quantiles.unsqueeze(1) - quantiles.unsqueeze(2)
    abs_pairwise_delta = torch.abs(pairwise_delta)
    huber_loss = torch.where(abs_pairwise_delta > 1, abs_pairwise_delta - 0.5, pairwise_delta**2 * 0.5)
    n_quantiles = quantiles.shape[1]
    tau = torch.arange(n_quantiles, device=quantiles.device).float() / n_quantiles + 1 / (2 * n_quantiles)
    loss = (torch.abs(tau.view(1, -1, 1) - (pairwise_delta.detach() < 0).float()) * huber_loss).mean()
    return loss

class DiscreteHERQuantileNetwork(nn.Module):
    def __init__(self, num_states, num_actions, num_quantiles, hidden_size=128):
        super(DiscreteHERQuantileNetwork, self).__init__()
        self.num_actions = num_actions
        self.num_quantiles = num_quantiles
        self.embedding = nn.Embedding(num_states, hidden_size)
        self.linear1 = nn.Linear(hidden_size * 2, hidden_size)
        self.linear2 = nn.Linear(hidden_size, num_actions * num_quantiles)

    def forward(self, state_idx, goal_idx):
        s_emb = self.embedding(state_idx)
        g_emb = self.embedding(goal_idx)
        x = torch.cat([s_emb, g_emb], dim=-1)
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x.view(-1, self.num_actions, self.num_quantiles)

class TQC_HER_Agent():
    def __init__(self, info):
        self.num_actions = info["num_actions"]
        self.num_states = info["num_states"]
        self.grid_width = info.get("grid_width", 10) 
        self.gamma = info.get("discount", 0.99)
        self.tau = 0.005
        self.lr = info.get("step_size", 0.001)
        self.batch_size = 64
        self.device = torch.device("cpu")
        self.num_quantiles = 25
        self.num_critics = 5
        self.top_quantiles_to_drop = 2
        self.k_future = 4

        # Auto Alpha
        self.target_entropy = -0.98 * np.log(1 / self.num_actions)
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optim = optim.Adam([self.log_alpha], lr=self.lr)
        self.alpha = self.log_alpha.exp()

        self.critics = nn.ModuleList([DiscreteHERQuantileNetwork(self.num_states, self.num_actions, self.num_quantiles).to(self.device) for _ in range(self.num_critics)])
        self.target_critics = nn.ModuleList([DiscreteHERQuantileNetwork(self.num_states, self.num_actions, self.num_quantiles).to(self.device) for _ in range(self.num_critics)])
        for i in range(self.num_critics): self.target_critics[i].load_state_dict(self.critics[i].state_dict())

        # Reuse Actor from SAC file or redefine similar structure
        from Sac_HER import DiscreteHERPolicyNetwork
        self.actor = DiscreteHERPolicyNetwork(self.num_states, self.num_actions).to(self.device)

        self.critic_optim = optim.Adam(self.critics.parameters(), lr=self.lr)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=self.lr)
        self.replay_buffer = [] 
        self.episode_buffer = [] 
        self.rand_generator = np.random.RandomState(info["seed"])

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
        return None

    def agent_end(self):
        for t, transition in enumerate(self.episode_buffer):
            self.replay_buffer.append(transition)
            state, action, _, next_state, _, _ = transition
            future_indices = range(t + 1, len(self.episode_buffer))
            if len(future_indices) > 0:
                count = min(self.k_future, len(future_indices))
                for idx in random.sample(future_indices, count):
                    achieved_goal = self.episode_buffer[idx][3]
                    new_reward = self.compute_reward(next_state, achieved_goal)
                    self.replay_buffer.append((state, action, new_reward, next_state, next_state==achieved_goal, achieved_goal))
        
        if len(self.replay_buffer) > 50000: self.replay_buffer = self.replay_buffer[-50000:]
        
        q_loss, a_loss = 0, 0
        updates = len(self.episode_buffer)
        for _ in range(updates):
            q, a = self.update_parameters()
            q_loss += q
            a_loss += a
        return q_loss/max(1, updates), a_loss/max(1, updates)

    def update_parameters(self):
        if len(self.replay_buffer) < self.batch_size: return 0, 0
        indices = self.rand_generator.randint(0, len(self.replay_buffer), size=self.batch_size)
        batch = [self.replay_buffer[i] for i in indices]
        state_b, action_b, reward_b, next_state_b, done_b, goal_b = zip(*batch)

        s = torch.LongTensor(state_b).to(self.device)
        a = torch.LongTensor(action_b).unsqueeze(1).to(self.device)
        r = torch.FloatTensor(reward_b).unsqueeze(1).to(self.device)
        ns = torch.LongTensor(next_state_b).to(self.device)
        d = torch.FloatTensor(done_b).unsqueeze(1).to(self.device)
        g = torch.LongTensor(goal_b).to(self.device)

        with torch.no_grad():
            next_probs, next_logs = self.actor(ns, g)
            
            # --- FIX: Target Calculation using Weighted Mixture ---
            # Get quantiles for ALL actions: [N_critics, Batch, N_actions, N_quantiles]
            target_qs = torch.stack([c(ns, g) for c in self.target_critics], dim=0)
            
            # We want to keep all atoms, weighted by their probability. 
            # Since atoms are unweighted in QR-DQN, we simulate this by sampling multiple actions 
            # OR (simpler/faster) sample ONE action per critic to maintain diversity.
            # Here we stick to the sampling fix: Sample actions based on policy
            
            # Broadcast next_probs to sample actions: [Batch]
            dist = torch.distributions.Categorical(next_probs)
            next_actions = dist.sample((self.num_critics,)) # [N_critics, Batch]
            
            # Gather atoms corresponding to these sampled actions
            # Expand actions to [N_critics, Batch, 1, N_quantiles]
            idx = next_actions.unsqueeze(2).unsqueeze(3).expand(-1, -1, 1, self.num_quantiles)
            chosen_qs = target_qs.gather(2, idx).squeeze(2) # [N_critics, Batch, N_quantiles]
            
            # Pooling and Truncation
            all_qs = chosen_qs.permute(1, 0, 2).reshape(self.batch_size, -1) # [Batch, N_crit*N_quant]
            sorted_qs, _ = torch.sort(all_qs, dim=1)
            keep = self.num_critics * self.num_quantiles - (self.top_quantiles_to_drop * self.num_critics)
            truncated_qs = sorted_qs[:, :keep]
            
            # Entropy Adjustment (Approximate using sampled log_prob)
            # log_pi for the sampled actions
            sampled_log_probs = next_logs.gather(1, next_actions.permute(1,0)).mean(dim=1, keepdim=True)
            target_dist = r + (1 - d) * self.gamma * (truncated_qs - self.alpha * sampled_log_probs)

        # Current quantiles
        current_qs_list = [c(s, g) for c in self.critics]
        a_expanded = a.unsqueeze(2).expand(-1, 1, self.num_quantiles)
        current_qs_taken = [q.gather(1, a_expanded).squeeze(1) for q in current_qs_list]
        
        q_loss = 0
        for cq in current_qs_taken:
            q_loss += quantile_huber_loss_f(cq, target_dist)
        
        self.critic_optim.zero_grad()
        q_loss.backward()
        self.critic_optim.step()

        # Actor Update
        probs, logs = self.actor(s, g)
        with torch.no_grad():
            # Mean of quantiles approximates Q-value
            q_means = []
            for c in self.critics:
                q_means.append(c(s, g).mean(dim=2))
            avg_q = torch.stack(q_means).mean(dim=0)
        
        actor_loss = (probs * (self.alpha * logs - avg_q)).sum(dim=1).mean()
        
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        # Alpha update
        alpha_loss = -(self.log_alpha * (logs + self.target_entropy).detach()).mean()
        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()
        self.alpha = self.log_alpha.exp()

        for i in range(self.num_critics):
            for t, p in zip(self.target_critics[i].parameters(), self.critics[i].parameters()):
                t.data.copy_(t.data * (1.0 - self.tau) + p.data * self.tau)

        return q_loss.item(), actor_loss.item()
    def save_model(self, filename="tqc_her_checkpoint"):
        torch.save({
            'actor': self.actor.state_dict(),
            'critics': self.critics.state_dict(), # Saves all 5 critics at once
            'log_alpha': self.log_alpha
        }, filename + ".pth")

    def load_model(self, filename="tqc_her_checkpoint"):
        if not os.path.exists(filename + ".pth"): return
        checkpoint = torch.load(filename + ".pth", map_location=self.device)
        
        # Load Actor and Alpha
        self.actor.load_state_dict(checkpoint['actor'])
        self.log_alpha = checkpoint['log_alpha']
        self.alpha = self.log_alpha.exp()
        
        # Load Critics (ModuleList handles the mapping automatically)
        self.critics.load_state_dict(checkpoint['critics'])
        
        # Sync Target Critics
        # We must copy the parameters from the loaded critics to the target critics
        for i in range(self.num_critics):
            self.target_critics[i].load_state_dict(self.critics[i].state_dict())