# Goal-Conditioned Reinforcement Learning with Hindsight Experience Replay

A PyTorch implementation of **SAC+HER** (Soft Actor-Critic with Hindsight Experience Replay) and **TQC+HER** (Truncated Quantile Critics with Hindsight Experience Replay) for goal-conditioned navigation in a 10×10 maze environment.

## Table of Contents

- [Overview](#overview)
- [Algorithms](#algorithms)
  - [Soft Actor-Critic (SAC)](#soft-actor-critic-sac)
  - [Truncated Quantile Critics (TQC)](#truncated-quantile-critics-tqc)
  - [Hindsight Experience Replay (HER)](#hindsight-experience-replay-her)
- [Environment](#environment)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Network Architecture](#network-architecture)
- [Results and Visualization](#results-and-visualization)
- [Technical Details](#technical-details)
- [Troubleshooting](#troubleshooting)
- [References](#references)

---

## Overview

This project implements two state-of-the-art off-policy reinforcement learning algorithms enhanced with Hindsight Experience Replay for sparse-reward, goal-conditioned tasks. The agent learns to navigate from a fixed starting position to a goal location in a maze with obstacles.

**Key Features:**
- Goal-conditioned policy and value networks with state-goal embedding concatenation
- HER with "future" strategy for sample-efficient learning in sparse reward settings
- Parallel training of both agents using Python's `multiprocessing`
- Discrete action space with learned embeddings (no one-hot encoding)
- Automatic checkpoint saving and training resumption
- Comprehensive logging and visualization tools

---

## Algorithms

### Soft Actor-Critic (SAC)

SAC is an off-policy actor-critic algorithm based on the maximum entropy reinforcement learning framework. It aims to maximize both the expected reward and the policy entropy:

$$J(\pi) = \sum_{t=0}^{T} \mathbb{E}_{(s_t, a_t) \sim \rho_\pi} \left[ r(s_t, a_t) + \alpha \mathcal{H}(\pi(\cdot | s_t)) \right]$$

**Key components in this implementation:**
- **Twin Critics**: Two Q-networks to reduce overestimation bias
- **Soft Value Target**: Uses minimum of twin Q-values minus entropy term
- **Discrete Action Space**: Outputs action probabilities via softmax, samples categorically
- **Fixed Temperature**: α = 0.2 (entropy coefficient)

### Truncated Quantile Critics (TQC)

TQC extends SAC with distributional reinforcement learning using quantile regression. Instead of estimating expected Q-values, it models the full return distribution.

**Key innovations:**
- **Multiple Quantile Critics**: 3 critics, each outputting 25 quantiles
- **Truncation Mechanism**: Drops top-k quantiles from the sorted atom set to control overestimation
- **Quantile Huber Loss**: Robust loss function for quantile regression

The truncation formula:
```
Total atoms = num_critics × num_quantiles = 3 × 25 = 75
Atoms to drop = top_quantiles_to_drop × num_critics = 2 × 3 = 6
Kept atoms = 75 - 6 = 69
```

### Hindsight Experience Replay (HER)

HER addresses the challenge of learning with sparse binary rewards by retroactively relabeling failed trajectories with achieved goals.

**"Future" Strategy Implementation:**
1. Store each transition in an episode buffer
2. At episode end, for each transition at time `t`:
   - Add the original transition to replay buffer
   - Sample `k=4` future states from times `t+1` to `T`
   - Relabel goal as the achieved state, recompute reward
   - Add these hindsight transitions to replay buffer

**Reward Function for HER:**
```python
def compute_reward(state_idx, goal_idx):
    if state_idx == goal_idx:
        return 1.0   # Goal reached
    return -0.1      # Step penalty
```

---

## Environment

The maze environment (`maze.py`) is a custom OpenAI Gym environment:

| Property | Value |
|----------|-------|
| Grid Size | 10 × 10 |
| State Space | 100 discrete states (row × 10 + col) |
| Action Space | 4 discrete actions (Up, Down, Right, Left) |
| Obstacle Density | 25% |
| Start Position | Bottom-left (9, 0) |
| Goal Position | Top-right (0, 9) |
| Random Seed | 42 (fixed maze layout) |

**Reward Structure (Original Environment):**
- `-1` for each step (including wall collisions)
- `0` for reaching the goal (episode terminates)

**Note:** The HER agents use a modified reward function (+1.0 for goal, -0.1 for steps) to better support hindsight relabeling.

---

## Project Structure

```
.
├── maze.py              # Gym environment for 10×10 maze navigation
├── Sac_HER.py           # SAC agent with HER implementation
├── TQC_HER.py           # TQC agent with HER implementation
├── Main_training.py     # Parallel training runner
├── Loss_Grapher.py      # Training metrics visualization
├── outputs/             # Generated during training
│   ├── training_sac_her_log_10x10.csv
│   ├── training_tqc_her_log_10x10.csv
│   ├── sac_her_10x10_model.pth
│   └── tqc_her_10x10_model.pth
└── README.md
```

---

## Installation

### Prerequisites

- Python 3.8+
- PyTorch 1.9+
- NumPy
- Pandas
- Matplotlib
- OpenAI Gym
- tqdm

### Setup

```bash
# Clone or download the project files
cd your_project_directory

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Install dependencies
pip install torch numpy pandas matplotlib gym tqdm
```

---

## Usage

### Training Both Agents in Parallel

```bash
python Main_training.py
```

This launches two parallel processes, one for SAC+HER and one for TQC+HER. Both agents train for 5000 episodes by default.

### Training a Single Agent

Modify `Main_training.py` or run directly:

```python
from Main_training import run_session

# Train only SAC
run_session("SAC", load_model=False, render_plots=False)

# Train only TQC
run_session("TQC", load_model=False, render_plots=False)
```

### Resuming Training

Set `load_model=True` (default) to resume from the last checkpoint:

```python
run_session("SAC", load_model=True, render_plots=False)
```

The training automatically detects the last episode from the CSV log and continues.

### Visualizing Results

```bash
python Loss_Grapher.py
```

This generates `comparison_plot_2.png` with side-by-side comparisons of:
- Average Reward per Episode
- Steps per Episode
- Critic Loss
- Actor Loss

### Testing the Environment

```bash
python maze.py
```

Runs a single episode with random actions to visualize the maze.

---

## Configuration

### Agent Hyperparameters

| Parameter | SAC+HER | TQC+HER | Description |
|-----------|---------|---------|-------------|
| Learning Rate | 0.001 | 0.0005 | Adam optimizer step size |
| Discount (γ) | 0.99 | 0.99 | Future reward discount factor |
| Soft Update (τ) | 0.005 | 0.005 | Target network update rate |
| Entropy (α) | 0.2 | 0.2 | Temperature parameter |
| Batch Size | 64 | 64 | Samples per update |
| Buffer Size | 50,000 | 50,000 | Replay buffer capacity |
| HER k_future | 4 | 4 | Hindsight goals per transition |
| Hidden Size | 128 | 128 | Network hidden layer dimension |

### TQC-Specific Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| num_quantiles | 25 | Atoms per critic |
| num_critics | 3 | Number of quantile critics |
| top_quantiles_to_drop | 2 | Dropped atoms per critic for truncation |

### Training Configuration

Located in `Main_training.py`:

```python
TRAIN_EPISODES = 5000    # Total training episodes
GOAL_STATE = (9, 9)      # Fixed goal (bottom-right)
RENDER_FREQUENCY = 1     # Render every N episodes (if enabled)
```

---

## Network Architecture

### Goal-Conditioned Policy Network

```
State Index ──► Embedding(100, 128) ──┐
                                      ├──► Concat ──► Linear(256, 128) ──► ReLU ──► Linear(128, 4) ──► Softmax ──► Action Probs
Goal Index  ──► Embedding(100, 128) ──┘
```

### Goal-Conditioned Q-Network (SAC)

```
State Index ──► Embedding(100, 128) ──┐
                                      ├──► Concat ──► Linear(256, 128) ──► ReLU ──► Linear(128, 4) ──► Q-values
Goal Index  ──► Embedding(100, 128) ──┘
```

### Goal-Conditioned Quantile Network (TQC)

```
State Index ──► Embedding(100, 128) ──┐
                                      ├──► Concat ──► Linear(256, 128) ──► ReLU ──► Linear(128, 100) ──► Reshape(4, 25) ──► Quantiles
Goal Index  ──► Embedding(100, 128) ──┘
```

The embedding layer learns a 128-dimensional representation for each of the 100 grid positions, allowing the network to capture spatial relationships.

---

## Results and Visualization

### Training Logs

CSV files contain per-episode metrics:

| Column | Description |
|--------|-------------|
| Episode | Episode number |
| Reward | Cumulative episode reward |
| Steps | Steps taken in episode |
| Critic_Loss | Average Q-network loss |
| Actor_Loss | Average policy loss |

### Expected Training Behavior

- **Early Training (0-500 episodes)**: High variance in rewards, exploration phase
- **Mid Training (500-2000 episodes)**: Gradual improvement, HER begins showing benefits
- **Late Training (2000+ episodes)**: Convergence toward optimal policy, reduced variance

### Comparison Metrics

The `Loss_Grapher.py` produces four subplots:
1. **Reward**: Higher is better; should trend toward 0 (minimal steps to goal)
2. **Steps**: Lower is better; optimal path length depends on maze layout
3. **Critic Loss**: Should stabilize; spikes indicate distribution shift
4. **Actor Loss**: Can be negative (entropy bonus); stability matters more than value

---

## Technical Details

### State Representation

States are converted to indices for embedding lookup:
```python
state_idx = grid_width * row + col  # e.g., (3, 5) → 3*10 + 5 = 35
```

### HER Episode Processing

```python
def agent_end(self):
    for t, transition in enumerate(episode_buffer):
        # Add original transition
        replay_buffer.append(transition)
        
        # Sample k future states as hindsight goals
        future_indices = range(t + 1, len(episode_buffer))
        for idx in random.sample(future_indices, min(k_future, len(future_indices))):
            achieved_goal = episode_buffer[idx].next_state
            new_reward = compute_reward(next_state, achieved_goal)
            new_done = (next_state == achieved_goal)
            replay_buffer.append((state, action, new_reward, next_state, new_done, achieved_goal))
```

### TQC Truncation Logic

```python
# Gather quantiles from all target critics
all_target_quantiles = concat([critic(next_state, goal) for critic in target_critics])

# Sort and truncate
sorted_quantiles = sort(all_target_quantiles, dim=1)
truncated = sorted_quantiles[:, :keep_atoms]  # Remove highest quantiles

# Use truncated distribution for target computation
target = reward + gamma * (truncated - alpha * log_prob)
```

---

## Troubleshooting

### Common Issues

**"No saved model found! Starting from scratch."**
- This is expected on first run. The checkpoint will be created after the first save interval (every 10 episodes).

**GUI crashes during parallel training**
- Ensure `render_plots=False` when running parallel processes. Matplotlib is not thread-safe.

**Memory usage grows over time**
- The replay buffer is capped at 50,000 transitions. If HER expansion causes issues, reduce `k_future` or buffer size.

**Training shows no improvement**
- Check that the maze is solvable (seed 42 should produce a valid maze)
- Verify goal state matches between `Main_training.py` and environment
- Ensure the reward function in agents matches expectations

**Slow training**
- Training runs on CPU by default. For GPU, modify `self.device = torch.device("cuda")` in agent files
- Reduce `TRAIN_EPISODES` for quick testing

### Debugging Tips

```python
# Check maze layout
env = Maze()
print(env.maze)  # 0 = empty, 1 = wall

# Verify state indexing
agent = SAC_HER_Agent(agent_info)
print(agent._get_idx((3, 5)))  # Should print 35

# Test single episode
state = env.reset()
action = agent.agent_start(state, (9, 9))
next_state, reward, done, _ = env.step(action)
```

---

## Supplementary: DQN Baseline Study

> **Note:** This section describes a supplementary experiment included in `HRL_project_all_live_viz.ipynb`. This notebook is **not the main project** but serves as a comparative baseline study to validate our choice of SAC and TQC over value-based methods like DQN.

### Motivation

To validate our algorithm selection, we implemented DQN-based methods on the more challenging **PointMaze_Large** environment from Gymnasium-Robotics. This study demonstrates why actor-critic methods (SAC, TQC) are fundamentally superior to value-based methods (DQN) for goal-conditioned navigation.

### Experiment Structure

The notebook implements a three-tier experimental framework:

| Tier | Method | Description |
|------|--------|-------------|
| **Tier 1** | DQN + Dense Reward | Baseline with distance-based reward shaping |
| **Tier 2** | DQN ± HER | Ablation study: sparse rewards with/without HER |
| **Tier 3** | HAC | Hierarchical Actor-Critic with subgoal decomposition |

### Environment: PointMaze_Large

Unlike the main project's 10×10 discrete maze, this study uses a continuous control environment:

| Property | Value |
|----------|-------|
| Environment | `PointMaze_Large-v3` |
| State Space | Continuous (position, velocity) |
| Action Space | Continuous 2D force → Discretized to 5 actions |
| Episode Limit | 500 steps |
| Success Threshold | 0.45 units from goal |

### Implementation Details

#### Discrete Action Wrapper

Maps 5 discrete actions to continuous force vectors:

```python
class DiscreteActionWrapper(gym.ActionWrapper):
    def __init__(self, env, n_actions=5, magnitude=10.0, repeat=5):
        self._action_map = {
            0: np.array([0.0, 0.0]),           # Stay
            1: np.array([0.0, magnitude]),     # Up
            2: np.array([0.0, -magnitude]),    # Down
            3: np.array([-magnitude, 0.0]),    # Left
            4: np.array([magnitude, 0.0]),     # Right
        }
        self.repeat = repeat  # Repeat action for momentum
```

#### Dense Reward Wrapper (Tier 1)

```python
class DenseRewardWrapper(gym.Wrapper):
    def step(self, action):
        # Reward = distance improvement + success bonus
        dense_reward = prev_distance - current_distance
        if info.get('is_success'):
            dense_reward += 20.0
        return obs, dense_reward, term, trunc, info
```

#### HER Integration (Tier 2)

Uses Stable-Baselines3's `HerReplayBuffer`:

```python
model = DQN(
    "MultiInputPolicy",
    train_env,
    replay_buffer_class=HerReplayBuffer,
    replay_buffer_kwargs={
        "n_sampled_goal": 8,
        "goal_selection_strategy": "future",
    },
)
```

### Results: Why DQN Fails

**All DQN-based methods achieved 0% success rate:**

| Method | Success Rate | Mean Steps | Key Finding |
|--------|-------------|------------|-------------|
| DQN + Dense Reward | 0.0% | 53.3 | Learns to approach but can't navigate maze |
| DQN without HER | 0.0% | 100.0 | Q-values collapse (no reward signal) |
| DQN with HER | 0.0% | 96.2 | HER provides signal but exploration fails |
| HAC | 0.0% | 88.4 | Hierarchical instability |

### Analysis: Why SAC/TQC Succeed Where DQN Fails

| Aspect | DQN (Failed) | SAC/TQC (Succeeded) |
|--------|--------------|---------------------|
| **Exploration** | ε-greedy (random, undirected) | Entropy maximization (diverse, goal-seeking) |
| **HER Synergy** | Poor (low trajectory diversity) | Strong (diverse trajectories to relabel) |
| **Policy** | Implicit (argmax Q) | Explicit (stochastic, smooth) |
| **Overestimation** | Target network only | Twin critics / Truncated quantiles |

**Key Insight:** HER's effectiveness depends on exploration quality. SAC's entropy-driven exploration generates diverse trajectories that HER can relabel into comprehensive training data. DQN's ε-greedy exploration produces nearly identical trajectories near the start, giving HER nothing useful to work with.

### Running the Supplementary Study

#### Google Colab (Recommended)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

1. Upload `HRL_project_all_live_viz.ipynb` to Colab
2. **Runtime → Restart runtime** (required for MuJoCo rendering)
3. Run all cells sequentially

#### Local Installation

```bash
# Additional dependencies for the notebook
pip install gymnasium[mujoco] gymnasium-robotics stable-baselines3 sb3-contrib
pip install tensorboard moviepy imageio pyvirtualdisplay

# For rendering (Linux)
sudo apt-get install xvfb ffmpeg libegl1-mesa-dev
```

#### Notebook Structure

```
Cells 0-10:  Setup (imports, config, wrappers, utilities)
Cells 11-13: Tier 1 - DQN with dense rewards + live visualization
Cells 14-16: Tier 2 - HER ablation + agent comparison
Cells 17-20: Tier 3 - HAC hierarchical learning
Cells 21-22: Results visualization and summary
```

### Live Training Visualization

The notebook features real-time training monitoring:

- **Live Loss Plots**: Watch training curves update during learning
- **Success Rate Tracking**: Monitor evaluation performance
- **Agent Visualization**: Observe trained agents navigate (or fail to navigate) the maze

```python
# Train with live visualization
model, eval_df = train_dqn_live(config, seed=42, plot_freq=500)

# Watch the trained agent
watch_agent_live(model, config, n_episodes=3)
```

### Conclusion

The complete failure of DQN-based methods validates our main project's choice of SAC+HER and TQC+HER:

1. **ε-greedy exploration is fundamentally inadequate** for complex navigation
2. **Entropy-based exploration (SAC/TQC) is essential** for HER effectiveness
3. **HER amplifies good exploration** but cannot create it from nothing
4. **Actor-critic methods provide faster credit assignment** than pure value-based methods

---

## Troubleshooting

### Common Issues

**"No saved model found! Starting from scratch."**
- This is expected on first run. The checkpoint will be created after the first save interval (every 10 episodes).

**GUI crashes during parallel training**
- Ensure `render_plots=False` when running parallel processes. Matplotlib is not thread-safe.

**Memory usage grows over time**
- The replay buffer is capped at 50,000 transitions. If HER expansion causes issues, reduce `k_future` or buffer size.

**Training shows no improvement**
- Check that the maze is solvable (seed 42 should produce a valid maze)
- Verify goal state matches between `Main_training.py` and environment
- Ensure the reward function in agents matches expectations

**Slow training**
- Training runs on CPU by default. For GPU, modify `self.device = torch.device("cuda")` in agent files
- Reduce `TRAIN_EPISODES` for quick testing

### Notebook-Specific Issues

**MuJoCo rendering error (`gladLoadGL error`)**
- Restart the runtime and run cells from the beginning
- Ensure environment variables are set before imports:
  ```python
  import os
  os.environ['MUJOCO_GL'] = 'egl'
  os.environ['PYOPENGL_PLATFORM'] = 'egl'
  ```

**Out of memory in Colab**
- Reduce `buffer_size` to 100,000
- Use `FAST_MODE = True` in configuration

### Debugging Tips

```python
# Check maze layout
env = Maze()
print(env.maze)  # 0 = empty, 1 = wall

# Verify state indexing
agent = SAC_HER_Agent(agent_info)
print(agent._get_idx((3, 5)))  # Should print 35

# Test single episode
state = env.reset()
action = agent.agent_start(state, (9, 9))
next_state, reward, done, _ = env.step(action)
```

---

## References

1. **SAC**: Haarnoja, T., et al. "Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor." *ICML*, 2018.

2. **TQC**: Kuznetsov, A., et al. "Controlling Overestimation Bias with Truncated Mixture of Continuous Distributional Quantile Critics." *ICML*, 2020.

3. **HER**: Andrychowicz, M., et al. "Hindsight Experience Replay." *NeurIPS*, 2017.

4. **DQN**: Mnih, V., et al. "Human-level control through deep reinforcement learning." *Nature*, 2015.

5. **HAC**: Levy, A., et al. "Learning Multi-Level Hierarchies with Hindsight." *ICLR*, 2019.

6. **Distributional RL**: Bellemare, M., et al. "A Distributional Perspective on Reinforcement Learning." *ICML*, 2017.

7. **Stable-Baselines3**: Raffin, A., et al. "Stable-Baselines3: Reliable Reinforcement Learning Implementations." *JMLR*, 2021.

8. **Gymnasium-Robotics**: Farama Foundation. [https://robotics.farama.org/](https://robotics.farama.org/)

---
