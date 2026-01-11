# -*- coding: utf-8 -*-
"""
Main_training.py
Updated to correctly synchronize dynamic goals (checkpoints) with HAC_HER.
"""
import numpy as np
import csv
import os
import pandas as pd
from maze import Maze
from tqdm import tqdm
from matplotlib import pyplot as plt
import torch
import random
import multiprocessing

try:
    from Sac_HER import SAC_HER_Agent
    from TQC_HER import TQC_HER_Agent
    from HAC_HER import HAC_HER_Agent 
except ImportError as e:
    print("Agent modules missing. Ensure Sac_HER.py, TQC_HER.py, and HAC_HER.py are present.")
    exit()

# --- CONFIGURATION ---
GRID_SIZE = 30   
TRAIN_EPISODES = 150
GLOBAL_SEED = 21 

# --- Control how many intermediate sub-rewards you want ---
NUM_CHECKPOINTS = 8  # Recommended: 4-8 for a 20x20 grid

# Dynamic Step Limit
MAX_EP_STEPS = GRID_SIZE * GRID_SIZE * 2
OUTPUT_FOLDER = f"worker_outputs_{GRID_SIZE}x{GRID_SIZE}_v2"

def set_global_seeds(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def run_session(agent_type, load_model=False, render_plots=False,record_model=True):
    set_global_seeds(GLOBAL_SEED)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    base_name = f"{GRID_SIZE}x{GRID_SIZE}_{agent_type}"
    save_filename = os.path.join(OUTPUT_FOLDER, f"model_{base_name}_checkpoint")
    log_filename = os.path.join(OUTPUT_FOLDER, f"log_{base_name}.csv")

    # Only HAC usually benefits from explicit sequential mode in the ENV,
    # but for fair comparison, if we use checkpoints, we use them for all.
    # However, standard practice: HAC needs it most.
    if agent_type == "HAC":
        active_checkpoints = NUM_CHECKPOINTS
        use_sequential = True
        print(f"[{agent_type}] Mode: Hierarchical Guided (Checkpoints: {active_checkpoints})")
    else:
        active_checkpoints = 0
        use_sequential = False
        print(f"[{agent_type}] Mode: Flat Sparse (Checkpoints: 0)")

    env = Maze(
        grid_size=GRID_SIZE, 
        seed=GLOBAL_SEED, 
        sequential_mode=use_sequential,
        num_checkpoints=active_checkpoints
    )

    
    valid_indices = env.get_valid_indices()

    if agent_type == "HAC": AgentClass = HAC_HER_Agent
    elif agent_type == "TQC": AgentClass = TQC_HER_Agent
    else: AgentClass = SAC_HER_Agent

    agent_info = {
        "num_actions": 4, 
        "num_states": GRID_SIZE * GRID_SIZE,
        "step_size": 0.001,
        "discount": 0.99, 
        "seed": GLOBAL_SEED,
        "grid_width": GRID_SIZE,
        "valid_indices": valid_indices
    }
    
    agent = AgentClass(agent_info)
    
    if load_model and os.path.exists(save_filename + ".pth"):
        agent.load_model(save_filename)
        
    if not os.path.exists(log_filename):
        with open(log_filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Episode", "Reward", "Steps", "Critic_Loss", "Actor_Loss"])

    print(f"[{agent_type}] Starting Training (Checkpoints: {NUM_CHECKPOINTS})...")

    for i in tqdm(range(TRAIN_EPISODES), desc=f"{agent_type}"):
        state = env.reset() 
        current_active_goal = env.goal
        
        # Start the agent
        result = agent.agent_start(state, current_active_goal)
        if isinstance(result, tuple): action, sub = result
        else: action, sub = result, None
        
        ep_rew = 0
        should_render = render_plots and (i % 50 == 0)

        if should_render: env.render(i, 0, sub)
        
        step_count = 0
        
        for t in range(MAX_EP_STEPS):
            next_state, reward, done, info = env.step(action)
            ep_rew += reward
            step_count += 1
            
            if should_render:
                env.render(i, t+1, sub)
            
            # Check if environment updated the goal (Checkpoint reached)
            new_active_goal = env.goal
            
            result = agent.agent_step(reward, next_state, done)
            if isinstance(result, tuple): action, sub = result
            else: action = result

            # --- CRITICAL FIX: SYNC GOAL UPDATE ---
            if new_active_goal != current_active_goal and not done:
                # 1. Update TQC/SAC (uses curr_goal_idx)
                if hasattr(agent, "curr_goal_idx"):
                    agent.curr_goal_idx = agent._get_idx(new_active_goal)
                
                # 2. Update HAC (uses final_goal_idx for Meta)
                if hasattr(agent, "final_goal_idx"):
                    agent.final_goal_idx = agent._get_idx(new_active_goal)
                    # Optional: Force Meta to re-evaluate subgoal immediately
                    # But HAC usually waits for subgoal_horizon. 
                    # We let HAC finish its current subgoal naturally.
                
                current_active_goal = new_active_goal
            # --------------------------------------
            
            if done: break
        
        # Agent End (HAC now loops internally here)
        q, a = agent.agent_end()

        with open(log_filename, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([i, ep_rew, step_count, q, a])

        #if i % 50 == 0:
            #agent.save_model(save_filename)
        if record_model:
            agent.save_model(save_filename)

    print(f"[{agent_type}] Training Complete.")

if __name__=='__main__':
    # Defaulting to HAC to test the new logic
    # Ensure render_plots=True if you want to see the GUI
    #run_session("HAC", load_model=True, render_plots=False,record_model=True)
    
    # --- Parallel Training Block (Uncomment to use) ---
    algorithms = ["HAC","SAC", "TQC"]
    processes = []
    print(f"Launching {len(algorithms)} parallel training sessions...")
    for algo in algorithms:
       p = multiprocessing.Process(target=run_session, args=(algo, True, False,True))
       processes.append(p)
       p.start()
    for p in processes:
       p.join()
    print("All training sessions finished!")