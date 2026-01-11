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
| SAC | Entropy-based | ✅ Good | General sparse-reward tasks |
| TQC | Entropy-based | ✅ Good | When faster convergence needed |
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

**Advantage over SAC**: Richer gradient signal from distributional learning.

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

## Discrete Grid Maze Environment

In addition to the continuous PointMaze environment, this repository includes implementations for **discrete grid-based maze navigation**. This provides a simpler testbed for understanding goal-conditioned RL algorithms before scaling to continuous control.

### Overview

The discrete environment represents mazes as 2D grids where:
- **State Space**: Integer grid position (x, y) or one-hot encoded state
- **Action Space**: 4 discrete actions (Up, Down, Left, Right)
- **Goal Space**: Target grid cell position
- **Reward**: Sparse (+1 on reaching goal, 0 or -0.1 otherwise)

This environment is particularly useful for:
- Rapid prototyping and debugging of RL algorithms
- Understanding HER behavior in a controlled setting
- Educational purposes and algorithm visualization
- Ablation studies without physics simulation overhead

### Discrete Environment Specifications

```
+---------------------------+-------------------+-------------------+
| Property                  | Small Maze (10x10)| Large Maze (20x20)|
+---------------------------+-------------------+-------------------+
| Grid Size                 | 10 x 10           | 20 x 20           |
| Number of States          | 100               | 400               |
| Number of Actions         | 4                 | 4                 |
| Optimal Path Length       | ~15-20 steps      | ~40-60 steps      |
| Max Episode Steps         | 100               | 500               |
| Wall Density              | ~20%              | ~25%              |
+---------------------------+-------------------+-------------------+
```

### Discrete File Structure

```
discrete/
|
+-- environments/
|   +-- grid_maze.py           # Core maze environment
|   +-- maze_layouts.py        # Predefined maze configurations
|   +-- wrappers.py            # Goal-conditioned wrapper
|
+-- agents/
|   +-- dqn_discrete.py        # DQN for discrete mazes
|   +-- dqn_her_discrete.py    # DQN + HER implementation
|   +-- sac_discrete.py        # SAC with discrete action head
|   +-- tqc_discrete.py        # TQC for discrete environments
|   +-- hac_discrete.py        # HAC with discrete low-level
|
+-- buffers/
|   +-- replay_buffer.py       # Standard experience replay
|   +-- her_buffer.py          # HER-augmented buffer
|   +-- hierarchical_buffer.py # HAC replay buffer
|
+-- utils/
|   +-- visualization.py       # Maze rendering and plotting
|   +-- metrics.py             # Evaluation utilities
|   +-- config.py              # Hyperparameter management
|
+-- experiments/
|   +-- run_discrete.py        # Main experiment runner
|   +-- ablations.py           # Ablation study scripts
|
+-- notebooks/
    +-- discrete_demo.ipynb    # Interactive demonstration
```

### Discrete Maze Environment Implementation

#### GridMazeEnv Class

```python
class GridMazeEnv(gym.Env):
    """
    Discrete grid maze environment for goal-conditioned RL.
    
    Observation Space:
        Dict with keys:
        - 'observation': Current position (2,) or one-hot (grid_size^2,)
        - 'achieved_goal': Current position (2,)
        - 'desired_goal': Target position (2,)
    
    Action Space:
        Discrete(4): 0=Up, 1=Down, 2=Left, 3=Right
    
    Reward:
        Sparse: +1 if goal reached, -0.1 step penalty (optional)
    """
    
    def __init__(
        self,
        grid_size: int = 10,
        maze_layout: str = 'random',
        use_one_hot: bool = False,
        step_penalty: float = 0.0,
        max_steps: int = 100
    ):
        ...
```

#### Maze Layouts

```python
MAZE_LAYOUTS = {
    'empty': """
        ..........
        ..........
        ..........
        ..........
        ..........
        ..........
        ..........
        ..........
        ..........
        ..........
    """,
    
    'simple_wall': """
        ..........
        ..........
        ..........
        ..........
        .####.....
        ..........
        ..........
        ..........
        ..........
        ..........
    """,
    
    'u_maze': """
        ..........
        .########.
        ..........
        ..........
        ..........
        ..........
        ..........
        .########.
        ..........
        ..........
    """,
    
    'four_rooms': """
        ....#.....
        ....#.....
        ....#.....
        ....#.....
        ##.###.###
        .....#....
        .....#....
        .....#....
        .....#....
        .....#....
    """,
}

# Legend: '.' = free cell, '#' = wall
```

### Discrete Algorithm Implementations

#### DQN for Discrete Mazes

```python
class DiscreteDQN:
    """
    Goal-conditioned DQN for discrete grid mazes.
    
    Architecture:
        State Embedding: grid_size^2 -> 64 (if one-hot)
                    or:  2 -> 64 (if coordinate)
        Goal Embedding:  2 -> 64
        Combined: 128 -> 256 -> 256 -> 4 (Q-values)
    """
    
    def __init__(
        self,
        state_dim: int,
        goal_dim: int = 2,
        hidden_dims: List[int] = [256, 256],
        embedding_dim: int = 64,
        learning_rate: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.995,
        target_update_freq: int = 100,
        device: str = 'cpu'
    ):
        ...
    
    def select_action(self, state, goal, deterministic=False):
        """Epsilon-greedy action selection."""
        ...
    
    def update(self, batch):
        """Standard DQN update with target network."""
        ...
```

#### HER Buffer for Discrete Environments

```python
class DiscreteHERBuffer:
    """
    Hindsight Experience Replay buffer for discrete mazes.
    
    Strategies:
        - 'future': Sample k states from future in episode
        - 'final': Use final achieved state as hindsight goal
        - 'episode': Sample randomly from episode
    """
    
    def __init__(
        self,
        buffer_size: int = 100000,
        her_k: int = 4,
        strategy: str = 'future',
        reward_fn: Callable = None
    ):
        ...
    
    def store_episode(self, episode: List[Transition]):
        """Store episode and generate HER samples."""
        for t, transition in enumerate(episode):
            # Store original transition
            self.buffer.append(transition)
            
            # Generate hindsight transitions
            if self.strategy == 'future':
                future_indices = np.random.choice(
                    range(t + 1, len(episode)),
                    size=min(self.her_k, len(episode) - t - 1),
                    replace=False
                )
                for idx in future_indices:
                    hindsight_goal = episode[idx].achieved_goal
                    hindsight_reward = self.reward_fn(
                        transition.next_state, hindsight_goal
                    )
                    self.buffer.append(Transition(
                        state=transition.state,
                        action=transition.action,
                        reward=hindsight_reward,
                        next_state=transition.next_state,
                        goal=hindsight_goal,
                        done=hindsight_reward > 0
                    ))
```

#### Discrete HAC Implementation

```python
class DiscreteHAC:
    """
    Hierarchical Actor-Critic for discrete grid mazes.
    
    High-Level: Outputs subgoal grid cells
    Low-Level: DQN selecting discrete actions to reach subgoal
    """
    
    def __init__(
        self,
        grid_size: int,
        subgoal_period: int = 5,
        high_level_lr: float = 1e-3,
        low_level_lr: float = 1e-3,
        use_her: bool = True,
        subgoal_test_prob: float = 0.3,
        device: str = 'cpu'
    ):
        # High-level: outputs subgoal as (x, y) coordinate
        self.high_policy = DiscreteSubgoalPolicy(
            state_dim=2,
            goal_dim=2,
            grid_size=grid_size,
            hidden_dims=[128, 128]
        )
        
        # Low-level: DQN to reach subgoals
        self.low_policy = DiscreteDQN(
            state_dim=2,
            goal_dim=2,  # subgoal
            hidden_dims=[128, 128]
        )
        
        self.subgoal_period = subgoal_period
        self.use_her = use_her
        ...
```

### Running Discrete Experiments

#### Quick Start

```python
from discrete.environments import GridMazeEnv
from discrete.agents import DiscreteDQN, DiscreteHERBuffer
from discrete.utils import DiscreteConfig

# Create environment
env = GridMazeEnv(
    grid_size=10,
    maze_layout='four_rooms',
    use_one_hot=False,
    max_steps=100
)

# Create agent
agent = DiscreteDQN(
    state_dim=2,
    goal_dim=2,
    hidden_dims=[256, 256]
)

# Create HER buffer
buffer = DiscreteHERBuffer(
    buffer_size=50000,
    her_k=4,
    strategy='future'
)

# Training loop
for episode in range(1000):
    obs, info = env.reset()
    episode_transitions = []
    done = False
    
    while not done:
        action = agent.select_action(
            obs['observation'],
            obs['desired_goal']
        )
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        episode_transitions.append(Transition(
            state=obs['observation'],
            action=action,
            reward=reward,
            next_state=next_obs['observation'],
            goal=obs['desired_goal'],
            achieved_goal=next_obs['achieved_goal'],
            done=done
        ))
        
        obs = next_obs
    
    # Store with HER augmentation
    buffer.store_episode(episode_transitions)
    
    # Update agent
    if len(buffer) > 256:
        batch = buffer.sample(256)
        agent.update(batch)
```

#### Command Line Interface

```bash
# Run DQN without HER on four_rooms maze
python discrete/experiments/run_discrete.py \
    --algorithm dqn \
    --maze four_rooms \
    --grid_size 10 \
    --total_episodes 5000 \
    --use_her false \
    --seed 42

# Run DQN with HER
python discrete/experiments/run_discrete.py \
    --algorithm dqn \
    --maze four_rooms \
    --grid_size 10 \
    --total_episodes 5000 \
    --use_her true \
    --her_k 4 \
    --seed 42

# Run HAC with HER on larger maze
python discrete/experiments/run_discrete.py \
    --algorithm hac \
    --maze u_maze \
    --grid_size 20 \
    --total_episodes 10000 \
    --use_her true \
    --subgoal_period 5 \
    --seed 42

# Run all ablations
python discrete/experiments/ablations.py \
    --output_dir results/discrete_ablations
```

### Discrete Configuration

```python
@dataclass
class DiscreteConfig:
    # Environment
    grid_size: int = 10
    maze_layout: str = 'four_rooms'
    use_one_hot: bool = False
    max_episode_steps: int = 100
    step_penalty: float = 0.0
    
    # Training
    total_episodes: int = 5000
    learning_rate: float = 1e-3
    buffer_size: int = 50000
    batch_size: int = 256
    gamma: float = 0.99
    
    # DQN-specific
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.995
    target_update_freq: int = 100
    
    # HER
    use_her: bool = True
    her_strategy: str = 'future'
    her_k: int = 4
    
    # HAC (hierarchical)
    subgoal_period: int = 5
    subgoal_test_prob: float = 0.3
    
    # Evaluation
    eval_freq: int = 100  # episodes
    n_eval_episodes: int = 50
    seeds: List[int] = field(default_factory=lambda: [42, 123, 456])
    
    # Paths
    log_dir: str = './logs/discrete'
    save_models: bool = True
```

### Discrete vs Continuous Comparison

| Aspect | Discrete Grid Maze | Continuous PointMaze |
|--------|-------------------|---------------------|
| State Representation | Grid coordinates or one-hot | Position + velocity (R^4) |
| Action Space | 4 discrete directions | Continuous force (R^2) |
| Transition Dynamics | Deterministic grid moves | Physics simulation |
| Computational Cost | Low (no physics) | High (MuJoCo) |
| Episode Length | 50-200 steps | 100-1000 steps |
| Training Time | Minutes | Hours |
| Debugging | Easy (visualize grid) | Harder (render videos) |
| Real-World Relevance | Limited | High (robot control) |

### Visualization for Discrete Mazes

```python
from discrete.utils.visualization import MazeVisualizer

viz = MazeVisualizer(env)

# Render single state
viz.render_state(
    agent_pos=(3, 4),
    goal_pos=(8, 8),
    path=[(3, 4), (4, 4), (5, 4), ...]
)

# Animate episode
viz.animate_episode(
    episode_data,
    save_path='episode.gif',
    fps=5
)

# Plot Q-value heatmap
viz.plot_q_values(
    agent,
    goal=(8, 8),
    action=0  # Up
)

# Plot visitation frequency
viz.plot_visitation_heatmap(
    visit_counts,
    title='State Visitation Frequency'
)
```

### Discrete Experiment Results

Expected performance on Four Rooms (10x10) maze:

| Algorithm | Success Rate | Mean Steps | Training Episodes |
|-----------|--------------|------------|-------------------|
| DQN (no HER) | 15-25% | 85+ | 5000 |
| DQN + HER | 75-85% | 35-45 | 5000 |
| SAC (discrete) | 30-40% | 70+ | 5000 |
| SAC + HER | 85-92% | 30-40 | 5000 |
| HAC (no HER) | 45-55% | 50-60 | 5000 |
| HAC + HER | 88-95% | 25-35 | 5000 |

Key observations from discrete experiments:
1. HER provides significant improvement across all algorithms
2. DQN benefits most from HER in discrete settings (unlike continuous)
3. HAC achieves best sample efficiency
4. Discrete environment allows faster iteration for hyperparameter tuning

### Troubleshooting Discrete Experiments

#### 1. Agent Stuck in Loops

```python
# Add exploration bonus to reward
reward = base_reward + 0.01 * novelty_bonus

# Or use intrinsic curiosity
from discrete.agents import ICMWrapper
agent = ICMWrapper(agent, curiosity_scale=0.1)
```

#### 2. HER Not Helping

```python
# Ensure episode stores achieved_goal correctly
assert 'achieved_goal' in transition
assert transition.achieved_goal is not None

# Check reward function
def compute_reward(achieved, desired, threshold=0.5):
    distance = np.linalg.norm(achieved - desired)
    return 1.0 if distance < threshold else 0.0
```

#### 3. Q-Values Exploding

```python
# Use gradient clipping
torch.nn.utils.clip_grad_norm_(agent.parameters(), max_norm=1.0)

# Or reduce learning rate
config.learning_rate = 1e-4
```

#### 4. Suboptimal Paths with HAC

```python
# Reduce subgoal period for denser supervision
config.subgoal_period = 3

# Increase subgoal testing probability
config.subgoal_test_prob = 0.5
```

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

