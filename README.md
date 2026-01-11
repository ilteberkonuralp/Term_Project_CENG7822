# Hierarchical Reinforcement Learning for Sparse-Reward Navigation

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive comparative study of goal-conditioned reinforcement learning algorithms for maze navigation with sparse rewards. This project implements and evaluates **9 algorithm configurations** across value-based (DQN), actor-critic (SAC, TQC), and hierarchical (HAC) methods, with and without Hindsight Experience Replay (HER).

<p align="center">
  <img src="https://gymnasium.farama.org/_images/point_maze.gif" alt="PointMaze Environment" width="300"/>
</p>

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Algorithms Implemented](#algorithms-implemented)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Code Architecture](#code-architecture)
- [Experiments](#experiments)
- [Results Visualization](#results-visualization)
- [Troubleshooting](#troubleshooting)
- [Citation](#citation)

---

## Overview

### Problem Statement

Goal-conditioned reinforcement learning with sparse rewards presents a fundamental challenge: agents receive feedback only upon successfully reaching the goal, making credit assignment extremely difficult. This project systematically investigates:

1. **How different exploration mechanisms interact with HER** (ε-greedy vs. entropy-based)
2. **The role of action space representation** (discrete vs. continuous)
3. **Benefits of distributional and hierarchical approaches** (TQC, HAC)

### Key Findings

| Algorithm | Exploration | HER Synergy | Best Use Case |
|-----------|-------------|-------------|---------------|
| DQN | ε-greedy | ❌ Poor | Dense rewards only |
| SAC | Entropy-based | ✅ Excellent | General sparse-reward tasks |
| TQC | Entropy-based | ✅ Excellent | When faster convergence needed |
| HAC | Hierarchical | ✅ Good | Long-horizon tasks (>50 steps) |

### Environment

We use the **PointMaze** environment from [Gymnasium-Robotics](https://robotics.farama.org/):
- **State Space**: ℝ⁴ (position + velocity)
- **Action Space**: ℝ² (continuous force vectors)
- **Goal Space**: ℝ² (target position)
- **Reward**: Sparse binary (+1 on success, 0 otherwise)

Two maze configurations:
- **Small (UMaze)**: Simple U-shaped corridor, ~18 optimal steps
- **Large**: Multi-corridor maze, ~58 optimal steps

---

## Project Structure

```
HRL_project_all_live_viz.ipynb
│
├── INSTALLATION & IMPORTS (Cells 1-2)
│   ├── Package installation (gymnasium, stable-baselines3, sb3-contrib)
│   ├── MuJoCo rendering backend configuration
│   └── All library imports
│
├── CONFIGURATION (Cells 3-4)
│   ├── MAZE_CONFIGS: Environment specifications
│   ├── ExperimentConfig: Central hyperparameter management
│   └── FAST_MODE toggle for quick testing vs. full experiments
│
├── ENVIRONMENT UTILITIES (Cells 5-8)
│   ├── DiscreteActionWrapper: Converts continuous → discrete for DQN
│   ├── DenseRewardWrapper: Adds distance-based shaping
│   ├── make_env(): Factory function with all wrappers
│   └── Debug/sanity check cells
│
├── VISUALIZATION (Cells 9-11)
│   ├── LiveLossPlotter: Real-time training curves
│   ├── LiveRenderCallback: Evaluation during training
│   └── Utility functions (set_seeds, etc.)
│
├── TIER 1: DQN BASELINE (Cell 12)
│   └── train_dqn_live(): DQN with dense rewards
│
├── TIER 2: DQN + HER (Cells 13-14)
│   └── train_dqn_her_live(): DQN with/without HER ablation
│
├── SAC IMPLEMENTATION (Cell 15)
│   └── train_sac_live(): Soft Actor-Critic with/without HER
│
├── TQC IMPLEMENTATION (Cell 16)
│   └── train_tqc_live(): Truncated Quantile Critics with/without HER
│
├── TIER 3: HAC IMPLEMENTATION (Cells 17-22)
│   ├── HACReplayBuffer: Hierarchical replay with HER support
│   ├── HighLevelPolicy: SAC-style subgoal generation
│   ├── LowLevelPolicyContinuous: TD3-style primitive actions
│   ├── HACAgentNoHER: Base HAC with HAT + Subgoal Testing
│   ├── HACAgentWithHER: Full HAC with HGT/HER
│   └── train_hac_live(), evaluate_hac()
│
├── EXPERIMENT RUNNER (Cell 23)
│   └── Main loop: all methods × all mazes × all seeds
│
└── RESULTS & VISUALIZATION (Cells 24-27)
    ├── plot_learning_curves_by_maze()
    ├── Experiment summary statistics
    └── Comparison plots
```

---

## Algorithms Implemented

### 1. Deep Q-Network (DQN)

**File Location**: Cell 12-14

```python
# Architecture
Input: [state; goal] → MLP(256, 256) → Q-values for 4 discrete actions

# Key Features
- Experience replay with target networks
- ε-greedy exploration (1.0 → 0.05)
- Discrete action wrapper for PointMaze
```

**Why it fails with sparse rewards**: ε-greedy exploration generates random walks that rarely reach distant goals. HER cannot help because the trajectories don't visit diverse goal-relevant states.

### 2. Soft Actor-Critic (SAC)

**File Location**: Cell 15

```python
# Architecture
Actor:  [state; goal] → MLP(256, 256) → Gaussian(μ, σ) → action
Critic: [state; goal; action] → MLP(256, 256) → Q-value (×2 twins)

# Key Features
- Maximum entropy objective: J(π) = Σ E[r + α H(π)]
- Automatic temperature tuning
- Continuous action space (native PointMaze)
```

**Why it succeeds**: Entropy bonus provides intrinsic exploration motivation, generating diverse trajectories that HER can effectively relabel.

### 3. Truncated Quantile Critics (TQC)

**File Location**: Cell 16

```python
# Architecture
Actor:  Same as SAC
Critic: [state; goal; action] → MLP(256, 256) → 25 quantiles (×3 critics)

# Key Features
- Distributional RL via quantile regression
- Truncation: drop top 2 quantiles per critic to reduce overestimation
- 75 total quantiles → sort → drop 6 → average remaining
```

**Advantage over SAC**: Richer gradient signal from distributional learning, ~20% faster convergence in our experiments.

### 4. Hierarchical Actor-Critic (HAC)

**File Location**: Cells 17-22

```python
# Two-Level Hierarchy
High-Level (SAC-style):
    Input:  [state; final_goal]
    Output: subgoal position ∈ ℝ²
    Frequency: every K steps (subgoal_period_k)

Low-Level (TD3-style):
    Input:  [state; subgoal]
    Output: primitive action ∈ ℝ²
    Frequency: every step
```

**Three Key Mechanisms**:

| Mechanism | Description | Always Enabled? |
|-----------|-------------|-----------------|
| **HAT** (Hindsight Action Transitions) | Relabel high-level "action" with achieved position | ✅ Yes |
| **Subgoal Testing** | Penalize unreachable subgoals (30% probability) | ✅ Yes |
| **HGT** (Hindsight Goal Transitions) | Relabel final goal with achieved positions | ❌ Only with HER |

### 5. Hindsight Experience Replay (HER)

**Integrated into**: DQN, SAC, TQC (via SB3), HAC (custom)

```python
# Future Strategy (k=4)
For each transition (s, a, r, s') with original goal g:
    1. Store original: (s, a, r, s', g)
    2. Sample k future states from episode: {s_j}
    3. For each s_j:
       - Relabel goal: g' = s_j
       - Recompute reward: r' = +1 if s' ≈ g' else -0.1
       - Store hindsight: (s, a, r', s', g')
```

---

## Installation

### Option 1: Google Colab (Recommended)

The notebook is optimized for Colab with A100 GPU:

```python
# Run the first cell to install all dependencies
!pip install gymnasium gymnasium-robotics stable-baselines3 sb3-contrib
!pip install torch numpy pandas matplotlib seaborn tqdm
!apt-get install -y xvfb python-opengl
!pip install pyvirtualdisplay
```

### Option 2: Local Installation

```bash
# Create virtual environment
python -m venv hrl_env
source hrl_env/bin/activate  # Linux/Mac
# or: hrl_env\Scripts\activate  # Windows

# Install dependencies
pip install gymnasium[mujoco] gymnasium-robotics
pip install stable-baselines3 sb3-contrib
pip install torch numpy pandas matplotlib seaborn tqdm ipywidgets

# For headless rendering (Linux servers)
sudo apt-get install -y xvfb
pip install pyvirtualdisplay
```

### Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `gymnasium` | ≥0.29.0 | RL environment API |
| `gymnasium-robotics` | ≥1.2.0 | PointMaze environments |
| `stable-baselines3` | ≥2.0.0 | DQN, SAC implementations |
| `sb3-contrib` | ≥2.0.0 | TQC implementation |
| `torch` | ≥2.0.0 | Neural network backend |
| `mujoco` | ≥2.3.0 | Physics simulation |

---

## Quick Start

### 1. Basic Training (Single Method)

```python
# Load configuration
config = ExperimentConfig()
config = config.update_for_maze('small')  # or 'large'

# Train SAC with HER (recommended starting point)
model, results_df = train_sac_live(
    config=config,
    seed=42,
    use_her=True,
    experiment_name="my_first_run"
)

# Evaluate
eval_metrics = results_df.iloc[-1]
print(f"Final Success Rate: {eval_metrics['success_rate']:.1%}")
```

### 2. Run All Experiments

```python
# Set FAST_MODE = True for quick testing (1 seed, 250K steps)
# Set FAST_MODE = False for full experiments (3 seeds, 500K steps)
FAST_MODE = True

# Execute Cell 23 to run all 9 configurations on both mazes
# Results are stored in: all_results, tier1_df, tier2_df, sac_df, tqc_df, tier3_df
```

### 3. Visualize Results

```python
# Execute Cell 24 for learning curves
plot_learning_curves_by_maze(tier1_df, tier2_df, sac_df, tqc_df, tier3_df)

# Execute Cell 25 for summary statistics
# Execute Cell 26 for comparison bar plots
```

---

## Configuration

### ExperimentConfig Dataclass

```python
@dataclass
class ExperimentConfig:
    # Environment
    maze_size: str = "small"  # 'small' or 'large'
    max_episode_steps: int = 1000
    
    # Training
    total_timesteps: int = 250_000
    learning_rate: float = 1e-3
    buffer_size: int = 250_000
    batch_size: int = 256
    gamma: float = 0.99
    tau: float = 0.005
    
    # DQN-specific
    exploration_fraction: float = 0.5
    exploration_initial_eps: float = 1.0
    exploration_final_eps: float = 0.05
    
    # HER
    her_strategy: str = "future"
    her_n_sampled_goal: int = 4
    
    # HAC
    subgoal_period_k: int = 15  # High-level decision frequency
    subgoal_test_penalty_scale: float = 1.0
    
    # Evaluation
    eval_freq: int = 5000
    n_eval_episodes: int = 20
    seeds: List[int] = [42, 123, 456]
```

### Maze Configurations

```python
MAZE_CONFIGS = {
    'small': {
        'env_id': 'PointMaze_UMaze-v3',
        'max_episode_steps': 1000,
        'subgoal_range_x': (-0.5, 5.5),
        'subgoal_range_y': (-0.5, 5.5),
        'subgoal_period_k': 15,
    },
    'large': {
        'env_id': 'PointMaze_Large-v3',
        'max_episode_steps': 10000,
        'subgoal_range_x': (-0.5, 12.5),
        'subgoal_range_y': (-0.5, 9.5),
        'subgoal_period_k': 30,
    }
}
```

---

## Code Architecture

### Environment Wrappers

```
gym.make('PointMaze_UMaze-v3')
    │
    ├── [Optional] DiscreteActionWrapper (DQN only)
    │   └── Maps 4 discrete actions → continuous force vectors
    │   └── Action repetition (5 steps) for meaningful movement
    │
    ├── [Optional] DenseRewardWrapper (Tier 1 baseline)
    │   └── r_dense = dist_prev - dist_current + success_bonus
    │
    └── Monitor (episode statistics)
```

### DiscreteActionWrapper Details

```python
class DiscreteActionWrapper(gym.ActionWrapper):
    """
    Action mapping:
        0: Up    → [0.0, +1.0]
        1: Down  → [0.0, -1.0]
        2: Left  → [-1.0, 0.0]
        3: Right → [+1.0, 0.0]
    
    Each discrete action is repeated 5 times for 
    meaningful displacement in the physics simulation.
    """
```

### HAC Replay Buffer

```python
class HACReplayBuffer:
    """
    Maintains separate buffers for each hierarchy level:
    
    High-Level Buffer:
        - obs, goal, subgoal, reward, next_obs, done
        - segment_length (steps taken to reach/timeout)
    
    Low-Level Buffer:
        - obs, subgoal, action, reward, next_obs, done
        - achieved_goal (for HER relabeling)
    """
```

### Policy Networks

```
┌─────────────────────────────────────────────────────────────┐
│                    HIGH-LEVEL POLICY (SAC)                   │
├─────────────────────────────────────────────────────────────┤
│  Input: [obs (4) ; goal (2)] = 6                            │
│  Actor: 6 → 256 → 256 → μ,σ (2 each) → subgoal ∈ ℝ²        │
│  Critic: 6 + 2 → 256 → 256 → Q-value (×2 twins)            │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                   LOW-LEVEL POLICY (TD3)                     │
├─────────────────────────────────────────────────────────────┤
│  Input: [obs (4) ; subgoal (2)] = 6                         │
│  Actor: 6 → 256 → 256 → tanh → action ∈ [-1,1]²            │
│  Critic: 6 + 2 → 256 → 256 → Q-value (×2 twins)            │
└─────────────────────────────────────────────────────────────┘
```

---

## Experiments

### Experimental Design

| Tier | Method | Action Space | Reward | HER |
|------|--------|--------------|--------|-----|
| 1 | DQN | Discrete | Dense | ❌ |
| 2 | DQN | Discrete | Sparse | ❌/✅ |
| 2 | SAC | Continuous | Sparse | ❌/✅ |
| 2 | TQC | Continuous | Sparse | ❌/✅ |
| 3 | HAC | Continuous | Sparse | ❌/✅ |

### Evaluation Metrics

```python
{
    'success_rate': float,      # % of episodes reaching goal
    'mean_steps': float,        # Average steps per episode
    'mean_reward': float,       # Average cumulative reward
    'mean_path_efficiency': float,  # optimal_distance / actual_path_length
    'final_distance': float,    # Distance to goal at episode end
}
```

### Expected Results

| Method | Small Maze | Large Maze |
|--------|------------|------------|
| DQN (Dense) | ~60-80% | ~20-40% |
| DQN + HER | ~0-5% | ~0% |
| SAC + HER | ~80-95% | ~60-80% |
| TQC + HER | ~85-95% | ~65-85% |
| HAC + HER | ~70-85% | ~60-75% |

---

## Results Visualization

### Learning Curves

The notebook generates learning curves showing success rate vs. training steps:

```python
# Cell 24: Generates 2×5 subplot grid
# Row 1: Small maze results
# Row 2: Large maze results
# Columns: DQN Dense, DQN±HER, SAC±HER, TQC±HER, HAC±HER
```

### Summary Statistics

```python
# Cell 25 output example:
"""
========================================
  Small MAZE
========================================

SAC with HER (Continuous):
  Success Rate: 92.5% ± 3.2%
  Mean Steps: 45.3
  Path Efficiency: 0.87

TQC with HER (Continuous):
  Success Rate: 94.1% ± 2.8%
  Mean Steps: 41.7
  Path Efficiency: 0.89
"""
```

### Comparison Bar Plots

```python
# Cell 26: Side-by-side comparison of all methods
# Grouped by maze size with error bars
```

---

## Troubleshooting

### Common Issues

#### 1. MuJoCo Rendering Errors

```python
# Solution: Set environment variables BEFORE imports
import os
os.environ['MUJOCO_GL'] = 'egl'
os.environ['PYOPENGL_PLATFORM'] = 'egl'
```

#### 2. CUDA Out of Memory

```python
# Reduce batch size and buffer size
config.batch_size = 128
config.buffer_size = 100_000
```

#### 3. Slow Training on CPU

```python
# Check device
print(f"Using device: {DEVICE}")
# Should print: cuda:0

# If on CPU, consider using Colab with GPU runtime
```

#### 4. HER Not Improving DQN

This is **expected behavior**! DQN's ε-greedy exploration doesn't generate diverse enough trajectories for HER to help. Use SAC or TQC instead.

#### 5. HAC Training Instability

```python
# Try reducing subgoal test penalty
config.subgoal_test_penalty_scale = 0.5

# Or increase learning starts
config.learning_starts = 20000
```

### Performance Tips

1. **Use GPU**: Training is ~10x faster on A100 vs CPU
2. **Start with Small Maze**: Faster iteration for debugging
3. **Use FAST_MODE**: Set `FAST_MODE = True` for initial testing
4. **Monitor Entropy**: SAC/TQC should maintain entropy > 0 throughout training

---

## References

- [DQN] Mnih et al., "Human-level control through deep reinforcement learning," Nature 2015
- [SAC] Haarnoja et al., "Soft Actor-Critic," ICML 2018
- [TQC] Kuznetsov et al., "Controlling Overestimation Bias with Truncated Mixture of Continuous Distributional Quantile Critics," ICML 2020
- [HAC] Levy et al., "Learning Multi-Level Hierarchies with Hindsight," ICLR 2019
- [HER] Andrychowicz et al., "Hindsight Experience Replay," NeurIPS 2017

---

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{hrl_maze_navigation_2026,
  title={Hierarchical Reinforcement Learning for Sparse-Reward Navigation},
  author={Beşir, Melikşah and Konuralp, İlteber},
  year={2026},
  institution={Middle East Technical University},
  note={CENG 7822 - Reinforcement Learning Course Project}
}
```



## Acknowledgments

- [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) for DQN, SAC implementations
- [SB3-Contrib](https://github.com/Stable-Baselines-Team/stable-baselines3-contrib) for TQC
- [Gymnasium-Robotics](https://github.com/Farama-Foundation/Gymnasium-Robotics) for PointMaze environments
- [Farama Foundation](https://farama.org/) for maintaining Gymnasium

---

