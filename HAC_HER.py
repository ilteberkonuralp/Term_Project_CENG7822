# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import random
from Sac_HER import DiscreteHERPolicyNetwork, DiscreteHERQNetwork # Reuse architectures

class HAC_HER_Agent():
    def __init__(self, info):
        self.num_actions = info["num_actions"]
        self.num_states = info["num_states"]
        self.grid_width = info.get("grid_width", 10)
        self.gamma = info.get("discount", 0.99)
        self.lr = info.get("step_size", 0.001)
        self.device = torch.device("cpu")
        
        # HAC Config
        self.subgoal_horizon = 10
        self.subgoal_test_rate = 0.3
        self.penalty_reward = -self.subgoal_horizon
        self.batch_size = 64
        
        # --- FIX: Valid Subgoal Mask ---
        valid_indices = info.get("valid_indices", None)
        if valid_indices is not None:
            self.valid_mask = torch.zeros(self.num_states, dtype=torch.bool).to(self.device)
            self.valid_mask[valid_indices] = True
        else:
            self.valid_mask = torch.ones(self.num_states, dtype=torch.bool).to(self.device)

        # --- Auto Alpha for Both Levels ---
        self.log_alpha_m = torch.zeros(1, requires_grad=True, device=self.device)
        self.optim_alpha_m = optim.Adam([self.log_alpha_m], lr=self.lr)
        self.alpha_m = self.log_alpha_m.exp()
        
        self.log_alpha_w = torch.zeros(1, requires_grad=True, device=self.device)
        self.optim_alpha_w = optim.Adam([self.log_alpha_w], lr=self.lr)
        self.alpha_w = self.log_alpha_w.exp()
        
        # Target entropies
        self.te_m = -0.98 * np.log(1 / self.num_states) # Meta explores states
        self.te_w = -0.98 * np.log(1 / self.num_actions)

        # META
        self.meta_actor = DiscreteHERPolicyNetwork(self.num_states, self.num_states).to(self.device)
        self.meta_critic1 = DiscreteHERQNetwork(self.num_states, self.num_states).to(self.device)
        self.meta_critic2 = DiscreteHERQNetwork(self.num_states, self.num_states).to(self.device)
        self.meta_critic1_target = DiscreteHERQNetwork(self.num_states, self.num_states).to(self.device)
        self.meta_critic2_target = DiscreteHERQNetwork(self.num_states, self.num_states).to(self.device)
        self.meta_critic1_target.load_state_dict(self.meta_critic1.state_dict())
        self.meta_critic2_target.load_state_dict(self.meta_critic2.state_dict())
        self.meta_actor_optim = optim.Adam(self.meta_actor.parameters(), lr=self.lr)
        self.meta_q_optim = optim.Adam(list(self.meta_critic1.parameters()) + list(self.meta_critic2.parameters()), lr=self.lr)

        # WORKER
        self.worker_actor = DiscreteHERPolicyNetwork(self.num_states, self.num_actions).to(self.device)
        self.worker_critic1 = DiscreteHERQNetwork(self.num_states, self.num_actions).to(self.device)
        self.worker_critic2 = DiscreteHERQNetwork(self.num_states, self.num_actions).to(self.device)
        self.worker_critic1_target = DiscreteHERQNetwork(self.num_states, self.num_actions).to(self.device)
        self.worker_critic2_target = DiscreteHERQNetwork(self.num_states, self.num_actions).to(self.device)
        self.worker_critic1_target.load_state_dict(self.worker_critic1.state_dict())
        self.worker_critic2_target.load_state_dict(self.worker_critic2.state_dict())
        self.worker_actor_optim = optim.Adam(self.worker_actor.parameters(), lr=self.lr)
        self.worker_q_optim = optim.Adam(list(self.worker_critic1.parameters()) + list(self.worker_critic2.parameters()), lr=self.lr)

        self.meta_buffer = []
        self.worker_buffer = []
        self.worker_ep_buffer = []
        self.subgoal_testing_log = []

    def _get_idx(self, state):
        if hasattr(state, '__len__'): return int(self.grid_width * state[0] + state[1])
        return int(state)
    
    def _get_coords(self, idx):
        return (int(idx // self.grid_width), int(idx % self.grid_width))

    def compute_reward(self, state_idx, goal_idx):
        return 0.0 if int(state_idx) == int(goal_idx) else -1.0

    def agent_start(self, state, final_goal):
        self.curr_state_idx = self._get_idx(state)
        self.final_goal_idx = self._get_idx(final_goal)
        self.worker_ep_buffer = []
        self.subgoal_testing_log = []
        self.is_testing_subgoal = (np.random.random() < self.subgoal_test_rate)
        self.subgoal_testing_log.append(self.is_testing_subgoal)
        self.curr_subgoal_idx = self._select_meta_action(self.curr_state_idx, self.final_goal_idx)
        self.subgoal_timer = 0
        action = self._select_worker_action(self.curr_state_idx, self.curr_subgoal_idx)
        self.prev_action = action
        return action, self._get_coords(self.curr_subgoal_idx)

    def agent_step(self, reward, next_state, done):
        next_state_idx = self._get_idx(next_state)
        worker_reward = self.compute_reward(next_state_idx, self.curr_subgoal_idx)
        worker_done = (next_state_idx == self.curr_subgoal_idx)
        
        # FIX: Append self.final_goal_idx to the tuple so we remember what the goal was
        self.worker_ep_buffer.append((
            self.curr_state_idx, 
            self.prev_action, 
            worker_reward, 
            next_state_idx, 
            worker_done, 
            self.curr_subgoal_idx, 
            self.gamma,
            self.final_goal_idx  # <--- NEW: Store the Goal ID
        ))      
        self.subgoal_timer += 1
        subgoal_finished = worker_done or (self.subgoal_timer >= self.subgoal_horizon)
        self.curr_state_idx = next_state_idx
        
        if done: return None, None

        if subgoal_finished:
            self.is_testing_subgoal = (np.random.random() < self.subgoal_test_rate)
            self.subgoal_testing_log.append(self.is_testing_subgoal)
            self.curr_subgoal_idx = self._select_meta_action(self.curr_state_idx, self.final_goal_idx)
            self.subgoal_timer = 0
            
        action = self._select_worker_action(self.curr_state_idx, self.curr_subgoal_idx)
        self.prev_action = action
        return action, self._get_coords(self.curr_subgoal_idx)

    def _select_meta_action(self, state_idx, goal_idx):
        s_t = torch.LongTensor([state_idx]).to(self.device)
        g_t = torch.LongTensor([goal_idx]).to(self.device)
        with torch.no_grad():
            probs, _ = self.meta_actor(s_t, g_t)
            masked_probs = probs * self.valid_mask.unsqueeze(0)
            if masked_probs.sum() == 0: masked_probs = self.valid_mask.float().unsqueeze(0)
            masked_probs = masked_probs / masked_probs.sum(dim=-1, keepdim=True)
            dist = torch.distributions.Categorical(masked_probs)
            subgoal = dist.sample()
        return subgoal.item()

    def _select_worker_action(self, state_idx, goal_idx):
        s_t = torch.LongTensor([state_idx]).to(self.device)
        g_t = torch.LongTensor([goal_idx]).to(self.device)
        with torch.no_grad():
            probs, _ = self.worker_actor(s_t, g_t)
            if self.is_testing_subgoal: action = torch.argmax(probs, dim=1)
            else: action = torch.distributions.Categorical(probs).sample()
        return action.item()

    def agent_end(self):
        # 1. Process Worker HER
        self._process_her(self.worker_ep_buffer, self.worker_buffer, 4)
        
        start_idx = 0
        subgoal_idx = 0
        meta_ep_buffer = []
        
        while start_idx < len(self.worker_ep_buffer):
            curr_sub = self.worker_ep_buffer[start_idx][5]
            
            # FIX: Retrieve the goal that was active during this specific segment
            # It is stored at index 7 (from Fix 1)
            segment_goal_idx = self.worker_ep_buffer[start_idx][7] 
            
            end_idx = start_idx
            for k in range(start_idx, len(self.worker_ep_buffer)):
                if self.worker_ep_buffer[k][5] != curr_sub: break
                end_idx = k + 1
            
            s_start = self.worker_ep_buffer[start_idx][0]
            s_end = self.worker_ep_buffer[end_idx-1][3]
            
            # FIX: Calculate reward using segment_goal_idx, NOT self.final_goal_idx
            r_meta = self.compute_reward(s_end, segment_goal_idx)
            done_meta = (s_end == segment_goal_idx)
            
            was_testing = self.subgoal_testing_log[subgoal_idx] if subgoal_idx < len(self.subgoal_testing_log) else False
            
            if was_testing:
                subgoal_achieved = (s_end == curr_sub)
                if not subgoal_achieved:
                    # FIX: Use segment_goal_idx in the tuple
                    meta_ep_buffer.append((s_start, curr_sub, float(self.penalty_reward), s_end, done_meta, segment_goal_idx, 0.0))
                else:
                    meta_ep_buffer.append((s_start, curr_sub, r_meta, s_end, done_meta, segment_goal_idx, self.gamma))
            else:
                # Hindsight Action Transition
                # FIX: Use segment_goal_idx in the tuple
                meta_ep_buffer.append((s_start, s_end, r_meta, s_end, done_meta, segment_goal_idx, self.gamma))
            
            start_idx = end_idx
            subgoal_idx += 1

        # 2. Process Meta HER
        self._process_her(meta_ep_buffer, self.meta_buffer, 4)

        updates = len(self.worker_ep_buffer)
        w_l = self._update_level(self.worker_buffer, self.worker_actor, self.worker_critic1, self.worker_critic2, self.worker_actor_optim, self.worker_q_optim, self.worker_critic1_target, self.worker_critic2_target, self.alpha_w, self.log_alpha_w, self.optim_alpha_w, self.te_w, updates)
        m_l = self._update_level(self.meta_buffer, self.meta_actor, self.meta_critic1, self.meta_critic2, self.meta_actor_optim, self.meta_q_optim, self.meta_critic1_target, self.meta_critic2_target, self.alpha_m, self.log_alpha_m, self.optim_alpha_m, self.te_m, len(meta_ep_buffer))
        
        return m_l[0] + w_l[0], m_l[1] + w_l[1]
    def _process_her(self, ep_trans, buffer, k):
        for t, trans in enumerate(ep_trans):
            # FIX: Only take the first 7 elements (Standard HER data) for the Worker training
            # We ignore the 8th element (parent goal) here because the Worker doesn't care about the Meta-Goal
            worker_trans = trans[:7] 
            buffer.append(worker_trans)
            
            future = range(t + 1, len(ep_trans))
            if future:
                for idx in random.sample(future, min(k, len(future))):
                    achieved = ep_trans[idx][3]
                    new_r = self.compute_reward(trans[3], achieved)
                    # Create HER transition (7 elements)
                    buffer.append((trans[0], trans[1], new_r, trans[3], trans[3]==achieved, achieved, self.gamma))
        if len(buffer) > 50000: del buffer[:-50000]

    def _update_level(self, buffer, actor, c1, c2, opt_actor, opt_c, t1, t2, alpha_tensor_stale, log_alpha, opt_alpha, te, updates):
        # Note: alpha_tensor_stale is the 'self.alpha_m' passed in, which we ignore 
        # because it is stale and connected to a freed graph. We compute a fresh one below.
        
        if len(buffer) < self.batch_size: return 0, 0
        q_acc, a_acc = 0, 0
        
        for _ in range(max(1, updates)):
            batch = random.sample(buffer, self.batch_size)
            s, a, r, ns, d, g, gam = zip(*batch)
            
            s = torch.LongTensor(s).to(self.device)
            a = torch.LongTensor(a).unsqueeze(1).to(self.device)
            r = torch.FloatTensor(r).unsqueeze(1).to(self.device)
            ns = torch.LongTensor(ns).to(self.device)
            d = torch.FloatTensor(d).unsqueeze(1).to(self.device)
            g = torch.LongTensor(g).to(self.device)
            gam_b = torch.FloatTensor(gam).unsqueeze(1).to(self.device)
            
            # --- FIX 1: Compute fresh alpha and detach for usage ---
            # We recompute alpha from the log_alpha parameter to get the current value.
            curr_alpha = log_alpha.exp()
            
            # --- CRITIC UPDATE ---
            with torch.no_grad():
                next_probs, next_logs = actor(ns, g)
                q1_n, q2_n = t1(ns, g), t2(ns, g)
                min_q = torch.min(q1_n, q2_n)
                
                # Use curr_alpha (inside no_grad, so it's safe)
                target_v = (next_probs * (min_q - curr_alpha * next_logs)).sum(dim=1, keepdim=True)
                target_q = r + (1-d) * gam_b * target_v
                target_q = torch.clamp(target_q, min=float(self.penalty_reward), max=0.0)

            q1 = c1(s, g).gather(1, a)
            q2 = c2(s, g).gather(1, a)
            q_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
            
            opt_c.zero_grad()
            q_loss.backward()
            opt_c.step()
            
            # --- ACTOR UPDATE ---
            probs, logs = actor(s, g)
            with torch.no_grad():
                q1_val, q2_val = c1(s, g), c2(s, g)
                min_q = torch.min(q1_val, q2_val)
            
            # --- FIX 2: Detach alpha here ---
            # We treat alpha as a constant for the Actor update. 
            # This prevents trying to backprop through the alpha graph (which caused your error).
            actor_loss = (probs * (curr_alpha.detach() * logs - min_q)).sum(dim=1).mean()
            
            opt_actor.zero_grad()
            actor_loss.backward()
            opt_actor.step()
            
            # --- ALPHA UPDATE ---
            # Optimize log_alpha based on the entropy target
            alpha_loss = -(log_alpha * (logs + te).detach()).mean()
            
            opt_alpha.zero_grad()
            alpha_loss.backward()
            opt_alpha.step()
            
            # Soft update
            for t, p in zip(t1.parameters(), c1.parameters()): t.data.copy_(t.data*0.995 + p.data*0.005)
            for t, p in zip(t2.parameters(), c2.parameters()): t.data.copy_(t.data*0.995 + p.data*0.005)
            
            q_acc += q_loss.item()
            a_acc += actor_loss.item()
            
        return q_acc/max(1, updates), a_acc/max(1, updates)
    def save_model(self, filename="hac_her_checkpoint"):
        torch.save({
            # --- Meta Level ---
            'meta_actor': self.meta_actor.state_dict(),
            'meta_critic1': self.meta_critic1.state_dict(),
            'meta_critic2': self.meta_critic2.state_dict(),
            'meta_log_alpha': self.log_alpha_m,  # FIXED: Changed from self.meta_log_alpha
            
            # --- Worker Level ---
            'worker_actor': self.worker_actor.state_dict(),
            'worker_critic1': self.worker_critic1.state_dict(),
            'worker_critic2': self.worker_critic2.state_dict(),
            'worker_log_alpha': self.log_alpha_w   # FIXED: Changed from self.worker_log_alpha
        }, filename + ".pth")

    def load_model(self, filename="hac_her_checkpoint"):
        if not os.path.exists(filename + ".pth"): return
        checkpoint = torch.load(filename + ".pth", map_location=self.device)
        
        # --- Load Meta Level ---
        self.meta_actor.load_state_dict(checkpoint['meta_actor'])
        self.meta_critic1.load_state_dict(checkpoint['meta_critic1'])
        self.meta_critic2.load_state_dict(checkpoint['meta_critic2'])
        
        # FIXED: Load data into the existing tensor so the optimizer stays linked
        self.log_alpha_m.data = checkpoint['meta_log_alpha'].data 
        self.alpha_m = self.log_alpha_m.exp()
        
        # Sync Meta Targets
        self.meta_critic1_target.load_state_dict(self.meta_critic1.state_dict())
        self.meta_critic2_target.load_state_dict(self.meta_critic2.state_dict())

        # --- Load Worker Level ---
        self.worker_actor.load_state_dict(checkpoint['worker_actor'])
        self.worker_critic1.load_state_dict(checkpoint['worker_critic1'])
        self.worker_critic2.load_state_dict(checkpoint['worker_critic2'])
        
        # FIXED: Load data into the existing tensor so the optimizer stays linked
        self.log_alpha_w.data = checkpoint['worker_log_alpha'].data
        self.alpha_w = self.log_alpha_w.exp()

        # Sync Worker Targets
        self.worker_critic1_target.load_state_dict(self.worker_critic1.state_dict())
        self.worker_critic2_target.load_state_dict(self.worker_critic2.state_dict())