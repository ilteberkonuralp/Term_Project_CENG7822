# -*- coding: utf-8 -*-
"""
training.py - Parallel Runner for SAC+HER and TQC+HER (10x10 Version)
"""

import numpy as np
import csv
import os
import pandas as pd
from maze import Maze
from tqdm import tqdm
from matplotlib import pyplot as plt
import torch
import multiprocessing # Added for parallel execution

# --- IMPORTS ---
try:
    from Sac_HER import SAC_HER_Agent
    from TQC_HER import TQC_HER_Agent
except ImportError as e:
    print(f"Error importing agents: {e}")
    print("Ensure Sac_HER.py and TQC_HER.py are in the same directory.")
    exit()

# --- SHARED CONFIGURATION ---
RENDER_FREQUENCY = 1
GOAL_STATE = (9, 9)   # <--- CHANGED: Fixed goal for 10x10 HER (Bottom-Right)
TRAIN_EPISODES = 5000  # How many episodes to train per agent

def run_session(agent_type, load_model=True, render_plots=False):
    """
    Runs a training session for a specific agent type.
    """
    
    # 1. Setup Agent Specifics
    if agent_type == "TQC":
        save_filename = "tqc_her_10x10_model"  # <--- CHANGED: Updated filename
        log_filename = "training_tqc_her_log_10x10.csv"
        learning_rate = 0.0005
        AgentClass = TQC_HER_Agent
        color = 'purple'
    else: # SAC
        save_filename = "sac_her_10x10_model"  # <--- CHANGED: Updated filename
        log_filename = "training_sac_her_log_10x10.csv"
        learning_rate = 0.001
        AgentClass = SAC_HER_Agent
        color = 'red'

    print(f"\n[{agent_type}] Starting Session...")
    print(f"[{agent_type}] Saving to: {save_filename}")
    
    agent_info = {
        "num_actions": 4, 
        "num_states": 100,      # <--- CHANGED: 10x10 = 100 states (was 900)
        "step_size": learning_rate,
        "discount": 0.99, 
        "seed": 42,
        "grid_width": 10        # <--- CHANGED: Grid width 10 (was 30)
    }
    
    env = Maze() 
    agent = AgentClass(agent_info)
    
    # 2. Load Model / Resume Logging
    start_episode = 0
    if load_model:
        agent.load_model(save_filename)
        if os.path.exists(log_filename):
            try:
                df = pd.read_csv(log_filename)
                if not df.empty:
                    start_episode = df["Episode"].iloc[-1] + 1
                    print(f"[{agent_type}] Resuming from Episode {start_episode}")
            except:
                pass

    # 3. Initialize CSV Header
    if not os.path.exists(log_filename):
        with open(log_filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Episode", "Reward", "Steps", "Critic_Loss", "Actor_Loss"])

    if render_plots:
        plt.ion()
    
    # 4. Training Loop
    # We use position argument in tqdm to prevent bars from overwriting each other in parallel
    pos = 0 if agent_type == "SAC" else 1
    
    for i in tqdm(range(start_episode, start_episode + TRAIN_EPISODES), desc=f"{agent_type}", position=pos):
        state = env.reset()
        
        # Pass the Goal to the agent (Required for HER)
        action = agent.agent_start(state, GOAL_STATE)
        
        ep_rew = 0
        should_render = render_plots and (i % RENDER_FREQUENCY == 0)

        if should_render:
            env.render(i, 0)
        
        # Episode Steps
        t = 0
        for t in range(1000):
            next_state, reward, done, info = env.step(action)
            ep_rew += reward
            
            if should_render:
                env.render(i, t+1)
                plt.pause(0.001)
            
            # Step the agent
            action = agent.agent_step(reward, next_state, done)
            
            if done:
                break
        
        # End of Episode: Updates
        avg_q_loss, avg_a_loss = agent.agent_end()

        # Log Data
        with open(log_filename, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([i, ep_rew, t+1, avg_q_loss, avg_a_loss])

        # Save Periodically
        if i % 10 == 0:
            agent.save_model(save_filename)

    print(f"\n[{agent_type}] Training Complete.")

    # 5. Final Plot (Only if not parallel, or handled carefully)
    if render_plots:
        visualize_log(log_filename, agent_type, color)

def visualize_log(filename, agent_name, color):
    # Visualization logic (Same as before)
    if os.path.exists(filename):
        try:
            plt.ioff()
            data = pd.read_csv(filename)
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))
            
            ax1.plot(data["Episode"], data["Reward"], alpha=0.3, color='blue')
            if len(data) > 20:
                rolling_mean = data["Reward"].rolling(window=20).mean()
                ax1.plot(data["Episode"], rolling_mean, color='darkblue', linewidth=2)
            ax1.set_title(f"Rewards ({agent_name})")

            ax2.plot(data["Episode"], data["Critic_Loss"], color=color, alpha=0.6)
            ax2.set_title(f"Critic Loss ({agent_name})")
            
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"Could not plot: {e}")

if __name__=='__main__':
    # We utilize multiprocessing to run both agents at the same time
    # NOTE: render_plots MUST be False for parallel runs to avoid GUI crashes
    
    print("Initializing Parallel Training...")

    # Create Process for SAC
    p1 = multiprocessing.Process(
        target=run_session, 
        args=("SAC", False, False) # Agent, Load_Model, Render_Plots
    )
    
    # Create Process for TQC
    p2 = multiprocessing.Process(
        target=run_session, 
        args=("TQC", False, False) # Agent, Load_Model, Render_Plots
    )

    # Start Processes
    p1.start()
    p2.start()

    # Wait for completion
    p1.join()
    p2.join()
    
    print("\nAll parallel training sessions completed.")