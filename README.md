# Hierarchical Reinforcement Learning for Sparse-Reward Navigation

 Hierarchical Reinforcement Learning (HRL) algorithm for solving sparse-reward navigation tasks using the **PointMaze** environment from Gymnasium-Robotics.

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Environment](#environment)
- [Algorithms Implemented](#algorithms-implemented)
- [Installation](#installation)
- [Usage](#usage)
- [Experimental Tiers](#experimental-tiers)
- [Key Components](#key-components)
- [Configuration](#configuration)
- [Results and Evaluation](#results-and-evaluation)
- [References](#references)

---

## Overview

This project investigates how different reinforcement learning techniques handle the **sparse reward problem** in goal-conditioned navigation tasks. The core challenge is that in sparse-reward settings, the agent only receives feedback upon reaching the goal, making credit assignment extremely difficult.

The project is organized into three progressive tiers, each building upon the previous:

| Tier | Method | Purpose |
|------|--------|---------|
| **Tier 1** | DQN with Dense Rewards | Establish a working baseline with discrete actions |
| **Tier 2** | DQN + HER | Demonstrate HER's effectiveness with sparse rewards |
| **Tier 3** | HAC (Hierarchical RL) | Enable long-horizon planning through temporal abstraction |

---

## Project Structure

```
HRL_project/
├── HRL_project.ipynb      # Main notebook with all implementations
├── logs/                  # Training logs and checkpoints
│   ├── tier1_*/           # Tier 1 experiment results
│   ├── tier2_*/           # Tier 2 experiment results
│   ├── tier3_*/           # Tier 3 experiment results
│   └── config.json        # Saved configuration
└── README.md              # This file
```

---

## Environment

### PointMaze (Gymnasium-Robotics)

The project uses the `PointMaze_Open-v3` environment where:

- **Agent**: A point mass that can move in 2D space
- **Goal**: Navigate to a target position in the maze
- **Observation Space**: Dictionary containing:
  - `observation`: Agent state (position + velocity) → shape `(4,)`
  - `achieved_goal`: Current agent position → shape `(2,)`
  - `desired_goal`: Target position → shape `(2,)`
- **Original Action Space**: Continuous 2D force vector
- **Modified Action Space**: Discrete 5 actions (up, down, left, right, stay)

### Discrete Action Wrapper

The continuous action space is converted to discrete actions for stability with DQN:

| Action ID | Direction | Force Vector |
|-----------|-----------|--------------|
| 0 | Stay | `[0.0, 0.0]` |
| 1 | Up (+Y) | `[0.0, 3.5]` |
| 2 | Down (-Y) | `[0.0, -3.5]` |
| 3 | Left (-X) | `[-3.5, 0.0]` |
| 4 | Right (+X) | `[3.5, 0.0]` |

Each action is repeated 5 times to ensure meaningful movement in the physics simulation.

---

## Algorithms Implemented

### 1. Deep Q-Network (DQN)

A value-based RL algorithm that learns to estimate the Q-function using neural networks.

**Key Features:**
- Experience replay buffer for sample efficiency
- Target network for training stability
- ε-greedy exploration with annealing
- Double DQN-style updates

### 2. Hindsight Experience Replay (HER)

A technique that enables learning from failures by relabeling goals in hindsight.

**Mechanism:**
- When an episode fails to reach the desired goal, HER creates additional training samples
- Failed trajectories are relabeled with the actually achieved state as the "goal"
- This converts sparse failures into dense learning signals

**Strategy Used:** `future` - samples goals from states visited later in the same episode

### 3. Hierarchical Actor-Critic (HAC)

A two-level hierarchical architecture for temporal abstraction:

**High-Level Policy (Manager):**
- Uses SAC (Soft Actor-Critic) for continuous subgoal generation
- Outputs 2D subgoal positions in the maze
- Operates every `k` timesteps (default: 30)

**Low-Level Policy (Worker):**
- Uses DQN for discrete action selection
- Receives subgoals from the high-level as temporary targets
- Executes primitive actions to reach subgoals

**Hindsight in HAC:**
Both levels use hindsight experience replay:
- Low-level: Relabels subgoals with achieved positions
- High-level: Relabels final goals with achieved positions

---

## Installation

### Requirements

```bash
pip install gymnasium[mujoco] gymnasium-robotics stable-baselines3 sb3-contrib tensorboard matplotlib pandas seaborn tqdm
```

### Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `gymnasium` | Latest | RL environment interface |
| `gymnasium-robotics` | Latest | PointMaze environments |
| `stable-baselines3` | Latest | DQN, replay buffers |
| `sb3-contrib` | Latest | HER implementation |
| `torch` | Latest | Neural network backend |
| `tensorboard` | Latest | Training visualization |

### Hardware

- **Recommended**: NVIDIA GPU (A100 used in development)
- **Minimum**: CPU-only training is possible but slower

---

## Usage

### Quick Start (Google Colab)

1. Open the notebook in Google Colab
2. Enable GPU runtime: `Runtime → Change runtime type → GPU`
3. Run all cells sequentially

### Running Experiments

```python
# Fast mode
FAST_MODE = True

# Full mode
FAST_MODE = False
```

### Training Individual Tiers

```python
# Tier 1: DQN Backbone
model, df = train_dqn_backbone(config, seed=42)

# Tier 2: DQN with HER
model, df = train_dqn_her(config, seed=42, use_her=True)

# Tier 3: HAC
agent, df = train_hac(config, seed=42)
```

---

## Experimental Tiers

### Tier 1: DQN Backbone

**Objective:** Establish a working baseline with discrete actions

**Setup:**
- Dense reward shaping: `reward = prev_distance - current_distance`
- Success bonus: +10 upon reaching goal
- This tier verifies the environment and action wrapper work correctly

**Expected Behavior:**
- Should learn to navigate with dense rewards
- Validates the discrete action transformation

### Tier 2: HER Ablation Study

**Objective:** Demonstrate HER's effectiveness with sparse rewards

**Experiments:**
1. **DQN without HER** (control): Expected to struggle or fail
2. **DQN with HER**: Expected to learn successfully

**Setup:**
- Pure sparse rewards: `reward = 0` (success) or `-1` (failure)
- Both conditions use identical architectures and hyperparameters
- This isolates HER's contribution

### Tier 3: HAC (Hierarchical RL)

**Objective:** Enable long-horizon planning through hierarchical decomposition

**Architecture:**
```
┌─────────────────────────────────────────────┐
│              High-Level Policy              │
│         (SAC: continuous subgoals)          │
│                    ↓                        │
│              Every k steps                  │
│                    ↓                        │
│        ┌─────────────────────┐              │
│        │   Subgoal (x, y)    │              │
│        └─────────────────────┘              │
│                    ↓                        │
│              Low-Level Policy               │
│       (DQN: discrete actions)               │
│                    ↓                        │
│        ┌─────────────────────┐              │
│        │  Environment Step   │              │
│        └─────────────────────┘              │
└─────────────────────────────────────────────┘
```

---

## Key Components

### DiscreteActionWrapper

Converts the continuous PointMaze action space to discrete actions:

```python
class DiscreteActionWrapper(gym.ActionWrapper):
    def __init__(self, env, n_actions=5, magnitude=3.5, repeat=5):
        # Actions: Stay, Up, Down, Left, Right
        # Each action is repeated 5 times for meaningful movement
```

### GoalManager

Manages reproducible train/test goal splits:

```python
class GoalManager:
    def __init__(self, env_id, n_train_goals=50, n_test_goals=20):
        # Generates fixed goal sets from environment resets
        # Ensures reproducibility across experiments
```

### DenseRewardWrapper

Adds reward shaping to accelerate learning:

```python
class DenseRewardWrapper(gym.Wrapper):
    def step(self, action):
        # reward = (previous_distance - current_distance) + success_bonus
```

### HACReplayBuffer

Separate replay buffers for hierarchical learning:

```python
class HACReplayBuffer:
    # High-level buffer: stores (obs, goal, subgoal, reward, next_obs, done)
    # Low-level buffer: stores (obs, subgoal, action, reward, next_obs, done)
```

### HighLevelPolicy

SAC-style policy for continuous subgoal generation:

```python
class HighLevelPolicy(nn.Module):
    # Actor: outputs mean and log_std for subgoal distribution
    # Critic: twin Q-networks for stable value estimation
    # Uses automatic entropy tuning
```

### LowLevelPolicy

DQN policy for discrete action selection:

```python
class LowLevelPolicy(nn.Module):
    # Q-network: outputs Q-values for each discrete action
    # Target network: for stable TD targets
```

---

## Configuration

### ExperimentConfig

All hyperparameters are centralized in a dataclass:

```python
@dataclass
class ExperimentConfig:
    # Environment
    maze_map: str = "PointMaze_Open-v3"
    max_episode_steps: int = 500
    n_discrete_actions: int = 5
    
    # Training
    total_timesteps: int = 750_000  # Fast mode
    learning_rate: float = 1e-3
    buffer_size: int = 500_000
    batch_size: int = 256
    gamma: float = 0.99
    
    # DQN-specific
    exploration_fraction: float = 0.5
    exploration_initial_eps: float = 1.0
    exploration_final_eps: float = 0.1
    
    # HER-specific
    her_strategy: str = "future"
    her_n_sampled_goal: int = 8
    
    # HAC-specific
    subgoal_period_k: int = 30
    subgoal_dim: int = 2
    subgoal_range: Tuple[float, float] = (0.0, 8.0)
    
    # Evaluation
    eval_freq: int = 50_000
    n_eval_episodes: int = 25
```

### Training Modes

| Mode | Timesteps | Seeds | Eval Episodes | Purpose |
|------|-----------|-------|---------------|---------|
| Fast | 750K | 1 | 25 | Progress report |
| Full | 1M | 3 | 50 | Final report |

---

## Results and Evaluation

### Metrics

| Metric | Description |
|--------|-------------|
| **Success Rate** | Fraction of episodes reaching the goal |
| **Mean Steps** | Average episode length |
| **Path Efficiency** | `goal_distance / actual_path_length` |
| **Steps to Success** | Steps in successful episodes only |

### Visualization

The notebook generates:
- Learning curves with confidence intervals
- HER ablation comparison plots
- All-methods comparison plots

### Expected Results

| Method | Success Rate | Notes |
|--------|--------------|-------|
| Tier 1: DQN (Dense) | ~0-30% | Baseline with reward shaping |
| Tier 2: DQN (Sparse, no HER) | ~0% | Demonstrates sparse reward difficulty |
| Tier 2: DQN + HER | ~20-60% | HER enables sparse reward learning |
| Tier 3: HAC | ~10-40% | Hierarchical planning |

*Note: Results vary with seeds and training duration*

---

## Key Implementation Details

### Physics Fix

The action repeat mechanism is critical for MuJoCo environments:

```python
def step(self, action):
    for _ in range(self.repeat):  # repeat=5
        obs, reward, term, trunc, info = self.env.step(cont_action)
        # Accumulate rewards, check termination
```

Without action repetition, single-step forces produce negligible movement.

### Hindsight Relabeling in HAC

Both levels benefit from hindsight:

```python
# High-level hindsight
hindsight_goal = next_achieved.copy()  # What we actually reached
hindsight_reward = 0.0  # Success by definition
buffer.add_high_transition(start_state, hindsight_goal, subgoal, ...)

# Low-level intrinsic reward
low_reward = -1.0 if distance_to_subgoal > threshold else 0.0
```

### Subgoal Period

The temporal abstraction ratio is controlled by `subgoal_period_k`:

- Too small (k=5): High-level learns nothing useful
- Too large (k=100): Low-level struggles to reach distant subgoals
- Default (k=30): Balanced trade-off

---

## References

### Papers

1. **DQN**: Mnih et al., "Human-level control through deep reinforcement learning" (2015)
2. **HER**: Andrychowicz et al., "Hindsight Experience Replay" (2017)
3. **HAC**: Levy et al., "Learning Multi-Level Hierarchies with Hindsight" (2019)
4. **SAC**: Haarnoja et al., "Soft Actor-Critic" (2018)

### Libraries

- [Gymnasium-Robotics](https://robotics.farama.org/)
- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/)
- [PyTorch](https://pytorch.org/)

---

## Troubleshooting

### Common Issues

1. **No movement in environment**
   - Ensure action magnitude is sufficient (default: 3.5)
   - Check action repeat is enabled (default: 5)

2. **CUDA out of memory**
   - Reduce `buffer_size` or `batch_size`
   - Use CPU training: `DEVICE = torch.device("cpu")`

3. **Zero success rate throughout training**
   - Normal for sparse reward without HER
   - Check goal positions are reachable
   - Verify dense reward wrapper is applied (Tier 1)

4. **HER not improving performance**
   - Ensure using sparse rewards (not dense)
   - Check `her_n_sampled_goal` parameter
   - Verify `her_strategy` is "future"

---

---

## Author
meliksah-besir
Created for Hierarchical Reinforcement Learning coursework.
