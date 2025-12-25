import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os



# Check if files exist, otherwise use dummy data
file_tqc = './outputs/training_tqc_her_log_10x10.csv'
file_sac = './outputs/training_sac_her_log_10x10.csv'


df_tqc = pd.read_csv(file_tqc).iloc[:200]
df_sac = pd.read_csv(file_sac).iloc[:200]

# Plotting
def plot_comparison(df1, name1, df2, name2):
    # Apply rolling window for smoother plots
    window = 20
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Comparison: {name1} vs {name2}', fontsize=16)
    
    # Reward
    axes[0, 0].plot(df1['Episode'], df1['Reward'], label=name1)
    axes[0, 0].plot(df2['Episode'], df2['Reward'], label=name2)
    axes[0, 0].set_title('Average Reward')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Steps
    axes[0, 1].plot(df1['Episode'], df1['Steps'], label=name1)
    axes[0, 1].plot(df2['Episode'], df2['Steps'], label=name2)
    axes[0, 1].set_title('Steps per Episode')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Steps')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # Critic Loss
    axes[1, 0].plot(df1['Episode'], df1['Critic_Loss'], label=name1)
    axes[1, 0].plot(df2['Episode'], df2['Critic_Loss'], label=name2)
    axes[1, 0].set_title('Critic Loss')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # Actor Loss
    axes[1, 1].plot(df1['Episode'], df1['Actor_Loss'], label=name1)
    axes[1, 1].plot(df2['Episode'], df2['Actor_Loss'], label=name2)
    axes[1, 1].set_title('Actor Loss')
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig('comparison_plot_2.png')
    print("Plot saved to comparison_plot_2.png")

plot_comparison(df_tqc, 'TQC+HER', df_sac, 'SAC+HER')