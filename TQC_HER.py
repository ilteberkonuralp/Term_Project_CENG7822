# -*- coding: utf-8 -*-
"""
TQC_HER.py - Truncated Quantile Critics with Hindsight Experience Replay
(Updated for 10x10 Grid)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import random

# --- Helper: Quantile Huber Loss ---
def quantile_huber_loss_f(quantiles, target_quantiles):
    """
    quantiles: (batch_size, num_quantiles)
    target_quantiles: (batch_size, num_quantiles_target)
    """
    pairwise_delta = target_quantiles.unsqueeze(1) - quantiles.unsqueeze(2)  # (B, n_q, n_target)
    abs_pairwise_delta = torch.abs(pairwise_delta)
    huber_loss = torch.where(abs_pairwise_delta > 1,
                             abs_pairwise_delta - 0.5,
                             pairwise_delta * pairwise_delta * 0.5)
    
    n_quantiles = quantiles.shape[1]
    tau = torch.arange(n_quantiles, device=quantiles.device).float() / n_quantiles + 1 / (2 * n_quantiles)
    loss = (torch.abs(tau.view(1, -1, 1) - (pairwise_delta.detach() < 0).float()) * huber_loss).mean()
    return loss

# --- Network Definitions ---

class DiscreteHERPolicyNetwork(nn.Module):
    def __init__(self, num_states, num_actions, hidden_size=128):
        super(DiscreteHERPolicyNetwork, self).__init__()
        self.embedding = nn.Embedding(num_states, hidden_size)
        # Input is State + Goal embeddings concatenated
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

class DiscreteHERQuantileNetwork(nn.Module):
    def __init__(self, num_states, num_actions, num_quantiles, hidden_size=128):
        super(DiscreteHERQuantileNetwork, self).__init__()
        self.num_actions = num_actions
        self.num_quantiles = num_quantiles
        
        self.embedding = nn.Embedding(num_states, hidden_size)
        self.linear1 = nn.Linear(hidden_size * 2, hidden_size)
        # Output is (num_actions * num_quantiles)
        self.linear2 = nn.Linear(hidden_size, num_actions * num_quantiles)

    def forward(self, state_idx, goal_idx):
        s_emb = self.embedding(state_idx)
        g_emb = self.embedding(goal_idx)
        x = torch.cat([s_emb, g_emb], dim=-1)
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        # Reshape to (Batch, Actions, Quantiles)
        return x.view(-1, self.num_actions, self.num_quantiles)

# --- TQC Agent with HER ---

class TQC_HER_Agent():
    def __init__(self, info):
        self.num_actions = info["num_actions"]
        self.num_states = info["num_states"]
        # Updated default to 10 to match your new environment
        self.grid_width = info.get("grid_width", 10) 
        self.gamma = info.get("discount", 0.99)
        self.tau = 0.005
        self.alpha = 0.2
        self.lr = info.get("step_size", 0.001)
        self.batch_size = 64
        self.memory_capacity = 50000 
        self.device = torch.device("cpu")
        
        # TQC Specifics
        self.num_quantiles = 25
        self.num_critics = 3  # TQC often uses more than 2
        self.top_quantiles_to_drop = 2 # Number of atoms to drop per critic * batch
        
        # HER Config
        self.k_future = 4

        # Initialize Critics
        self.critics = nn.ModuleList([
            DiscreteHERQuantileNetwork(self.num_states, self.num_actions, self.num_quantiles).to(self.device)
            for _ in range(self.num_critics)
        ])
        self.target_critics = nn.ModuleList([
            DiscreteHERQuantileNetwork(self.num_states, self.num_actions, self.num_quantiles).to(self.device)
            for _ in range(self.num_critics)
        ])
        
        for i in range(self.num_critics):
            self.target_critics[i].load_state_dict(self.critics[i].state_dict())

        self.actor = DiscreteHERPolicyNetwork(self.num_states, self.num_actions).to(self.device)

        # Optimizers
        self.critic_optim = optim.Adam(self.critics.parameters(), lr=self.lr)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=self.lr)

        self.replay_buffer = [] 
        self.episode_buffer = [] 
        self.rand_generator = np.random.RandomState(info["seed"])

    def save_model(self, filename="tqc_her_checkpoint"):
        torch.save({
            'actor': self.actor.state_dict(),
            'critics': self.critics.state_dict(),
            'critic_optim': self.critic_optim.state_dict(),
            'actor_optim': self.actor_optim.state_dict(),
        }, filename + ".pth")
        print(f"Model saved to {filename}.pth")

    def load_model(self, filename="tqc_her_checkpoint"):
        if not os.path.exists(filename + ".pth"):
            print("No saved model found! Starting from scratch.")
            return
        checkpoint = torch.load(filename + ".pth", map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critics.load_state_dict(checkpoint['critics'])
        self.critic_optim.load_state_dict(checkpoint['critic_optim'])
        self.actor_optim.load_state_dict(checkpoint['actor_optim'])
        
        for i in range(self.num_critics):
            self.target_critics[i].load_state_dict(self.critics[i].state_dict())
            
        print(f"Model loaded successfully from {filename}.pth")

    def _get_idx(self, state):
        if hasattr(state, '__len__'):
             return int(self.grid_width * state[0] + state[1])
        return int(state)

    def compute_reward(self, state_idx, goal_idx):
        if state_idx == goal_idx:
            return 1.0
        return -0.1

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
        # 1. HER Replay Processing
        
        for t, transition in enumerate(self.episode_buffer):
            self.replay_buffer.append(transition)
            state, action, _, next_state, _, original_goal = transition
            
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
            pop_amt = len(self.replay_buffer) - self.memory_capacity
            self.replay_buffer = self.replay_buffer[pop_amt:]

        # 2. Batch Updates
        q_loss_sum, a_loss_sum = 0, 0
        updates_to_run = len(self.episode_buffer)
        
        for _ in range(updates_to_run):
             q, a = self.update_parameters()
             q_loss_sum += q
             a_loss_sum += a
             
        return q_loss_sum / updates_to_run, a_loss_sum / updates_to_run

    def update_parameters(self):
        if len(self.replay_buffer) < self.batch_size:
            return 0, 0

        indices = self.rand_generator.randint(0, len(self.replay_buffer), size=self.batch_size)
        batch = [self.replay_buffer[i] for i in indices]
        
        state_b, action_b, reward_b, next_state_b, done_b, goal_b = zip(*batch)

        state_batch = torch.LongTensor(state_b).to(self.device)
        action_batch = torch.LongTensor(action_b).unsqueeze(1).to(self.device)
        reward_batch = torch.FloatTensor(reward_b).unsqueeze(1).to(self.device)
        next_state_batch = torch.LongTensor(next_state_b).to(self.device)
        done_batch = torch.FloatTensor(done_b).unsqueeze(1).to(self.device)
        goal_batch = torch.LongTensor(goal_b).to(self.device)

        # --- 1. Critic Update (TQC Logic) ---
        
        with torch.no_grad():
            # Get next action probabilities
            next_probs, next_log_probs = self.actor(next_state_batch, goal_batch)
            
            # Get quantiles from all target critics
            # Shape: [n_critics, batch, n_actions, n_quantiles]
            target_quantiles_list = [
                critic(next_state_batch, goal_batch) for critic in self.target_critics
            ]
            
            # We want the quantiles for the ACTIONS sampled by the policy? 
            # In SAC Discrete, we usually expect Sum(prob * Q). 
            # For Distributional, we often just take the quantiles of the best action or sample.
            # Here we approximate: We mix the distributions based on probs (complex) OR
            # Simplification: Assume the actor picks the best action (greedy) or sample one.
            # Let's sample an action from the next policy for the target (standard for SAC).
            
            next_dist = torch.distributions.Categorical(next_probs)
            next_action = next_dist.sample() # (Batch,)
            next_action_unsqueezed = next_action.unsqueeze(1).unsqueeze(2).expand(-1, 1, self.num_quantiles)

            # Gather quantiles for the chosen next action
            # Result list of: (Batch, 1, n_quantiles)
            chosen_quantiles = [
                q.gather(1, next_action_unsqueezed).squeeze(1) 
                for q in target_quantiles_list
            ]
            
            # Concatenate all atoms from all critics: (Batch, n_critics * n_quantiles)
            all_target_quantiles = torch.cat(chosen_quantiles, dim=1)
            
            # Sort ascending
            sorted_quantiles, _ = torch.sort(all_target_quantiles, dim=1)
            
            # TRUNCATION: Drop the top k atoms (overestimation control)
            total_atoms = self.num_critics * self.num_quantiles
            keep_atoms = total_atoms - (self.top_quantiles_to_drop * self.num_critics)
            truncated_quantiles = sorted_quantiles[:, :keep_atoms]
            
            # Calculate Target Y
            # Entropy term needs to be added to the value, typically subtracted from target
            # For quantile regression, we apply it to the scalar reward usually.
            # Y = r + gamma * (Q_t - alpha * log_pi)
            
            # Entropy adjustment
            # We subtract alpha * log_prob of the selected action from the atoms
            log_prob_next = next_log_probs.gather(1, next_action.unsqueeze(1))
            entropy_term = self.alpha * log_prob_next
            
            target_dist = reward_batch + (1 - done_batch) * self.gamma * (truncated_quantiles - entropy_term)

        # Current Quantiles
        # Shape: (Batch, n_actions, n_quantiles)
        current_quantiles_list = [critic(state_batch, goal_batch) for critic in self.critics]
        
        # Select quantiles for the action actually taken
        # Shape: (Batch, n_quantiles)
        action_unsqueezed = action_batch.unsqueeze(2).expand(-1, 1, self.num_quantiles)
        current_quantiles_taken = [
            q.gather(1, action_unsqueezed).squeeze(1) 
            for q in current_quantiles_list
        ]
        
        # Compute Loss (Sum over all critics)
        q_loss = 0
        for current_q in current_quantiles_taken:
            q_loss += quantile_huber_loss_f(current_q, target_dist)

        self.critic_optim.zero_grad()
        q_loss.backward()
        self.critic_optim.step()

        # --- 2. Actor Update ---
        # We want to maximize expected Q
        # In TQC/Distributional, Expected Q is the mean of the quantiles
        probs, log_probs = self.actor(state_batch, goal_batch)
        
        with torch.no_grad():
            # Get Q-values (mean of quantiles) from first critic (or avg of all) to guide actor
            # Let's use average of all critics' means for robustness
            q_means_list = []
            for critic in self.critics:
                q_dist = critic(state_batch, goal_batch) # (B, Actions, Quantiles)
                q_means_list.append(q_dist.mean(dim=2)) # (B, Actions)
            
            avg_q_values = torch.stack(q_means_list).mean(dim=0) # (B, Actions)
            
        # SAC Actor Loss
        actor_loss = (probs * (self.alpha * log_probs - avg_q_values)).sum(dim=1).mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        # --- 3. Soft Update Targets ---
        for i in range(self.num_critics):
            for target, param in zip(self.target_critics[i].parameters(), self.critics[i].parameters()):
                target.data.copy_(target.data * (1.0 - self.tau) + param.data * self.tau)
            
        return q_loss.item(), actor_loss.item()