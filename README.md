---

```markdown
# Hierarchical Reinforcement Learning & HER Ablation Studies

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![Gymnasium](https://img.shields.io/badge/Gymnasium-Robotics-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

## 📖 Overview

This project provides a comprehensive implementation and comparative analysis of Reinforcement Learning algorithms on sparse-reward navigation tasks (specifically `PointMaze_UMaze-v3`). 

The study is structured into three tiers of increasing complexity, moving from standard off-policy baselines to Hindsight Experience Replay (HER) ablations, and finally to a custom implementation of **Hierarchical Actor-Critic (HAC)**. The goal is to demonstrate the necessity of goal-conditioned learning and temporal abstraction for solving long-horizon robotic tasks.

## 🚀 Key Features

* **Custom HAC Implementation:** A ground-up implementation of Hierarchical Actor-Critic with a two-level hierarchy (High-level subgoal generation, Low-level action execution).
* **Rigorous Evaluation:** Implements a `GoalManager` to ensure fixed train/test goal splits across seeds for statistically significant results.
* **Comprehensive Metrics:** Tracks Success Rate, Mean Steps, Path Efficiency, and Goal Distance.
* **Ablation Studies:** Includes dedicated experiments for Subgoal Period ($K$) and HER goal-relabeling impact.
* **Reproducibility:** Centralized `ExperimentConfig` dataclass to manage all hyperparameters.

---

## 🔬 Methodology & Architecture

The project is divided into three experimental tiers:

### Tier 1: Backbone Baselines (SAC vs. TQC)
We establish a performance baseline using **Soft Actor-Critic (SAC)** and **Truncated Quantile Critics (TQC)**.
* **Objective:** Compare standard maximum entropy RL against distributional RL in a continuous control setting.
* **Outcome:** Evaluates if distributional critics provide stability improvements in maze navigation.

### Tier 2: The Role of Hindsight (HER Ablation)
We investigate the impact of **Hindsight Experience Replay (HER)** on learning efficiency.
* **Setup:** Training SAC with and without HER relabeling.
* **Significance:** In sparse reward environments (like PointMaze), standard RL fails because the agent rarely encounters a reward. HER allows the agent to learn from failure by pretending the state it *did* reach was the intended goal.

### Tier 3: Hierarchical Actor-Critic (HAC)
A custom implementation of a two-layer hierarchical agent:
1.  **High-Level Policy:** Observes the state and final goal $\to$ Outputs a **Subgoal**.
2.  **Low-Level Policy:** Observes the state and the Subgoal $\to$ Outputs a primitive **Action**.
3.  **Hindsight at all levels:** Both levels utilize HER to learn from missed subgoals and missed final goals.

---

## 🛠️ Installation

This project requires Python 3.8+ and the MuJoCo physics engine.

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/yourusername/hac-pointmaze-rl.git](https://github.com/yourusername/hac-pointmaze-rl.git)
   cd hac-pointmaze-rl

```

2. **Install dependencies:**
It is recommended to use a virtual environment.
```bash
pip install gymnasium[mujoco] gymnasium-robotics stable-baselines3 sb3-contrib tensorboard matplotlib pandas seaborn tqdm

```


*Note: If running on Colab, the script includes a cell to install these automatically.*

---

## 💻 Usage

The entire experimental pipeline is contained within `project_codes.py`. The script is designed to run all three tiers sequentially.

### Configuration

You can modify hyperparameters in the `ExperimentConfig` dataclass at the top of the file:

```python
@dataclass
class ExperimentConfig:
    maze_map: str = "PointMaze_UMaze-v3"
    total_timesteps: int = 500_000
    subgoal_period_k: int = 10  # Frequency of high-level decisions
    her_strategy: str = "future"
    # ...

```

### Running Experiments

Run the main script:

```bash
python project_codes.py

```

This will:

1. Initialize the environments.
2. Run Tier 1 (SAC vs TQC).
3. Run Tier 2 (HER Ablation).
4. Run Tier 3 (HAC Training).
5. Generate comparisons and save plots to the `./logs` directory.

---

## 📊 Experimental Results & Visualizations

### 1. Learning Curves (Success Rate)

*Comparison of SAC, TQC, and HAC performance over 500k timesteps.*

> [Place your `learning_curves_success.png` here]

### 2. Path Efficiency Analysis

*Analysis of how optimal the paths taken by different agents are (Goal Distance / Path Length).*

> [Place your `learning_curves_efficiency.png` here]

### 3. HER Ablation Study

*Demonstrating the necessity of Hindsight Experience Replay in sparse reward settings.*

> [Place your `her_ablation.png` here]

### 4. Final Performance Comparison

*Bar chart comparing the final converged success rates across all methods.*

> [Place your `final_comparison.png` here]

---

## 📂 File Structure

```text
├── project_codes.py       # Main entry point containing all logic and classes
├── logs/                  # Generated during training
│   ├── experiment_summary.txt
│   ├── results_summary.csv
│   └── [plots and model checkpoints]
└── README.md              # Project documentation

```

## 📚 References

1. **SAC:** Haarnoja, T., et al. (2018). *Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor.*
2. **HER:** Andrychowicz, M., et al. (2017). *Hindsight Experience Replay.*
3. **HAC:** Levy, A., et al. (2019). *Learning Multi-Level Hierarchies with Hindsight.*
4. **TQC:** Kuznetsov, A., et al. (2020). *Controlling Overestimation Bias with Truncated Mixture of Continuous Distributional Quantile Critics.*

```

### How to use the Image Area:

1.  Run your code locally or in Colab.
2.  Go to the `logs/` folder created by the script.
3.  Locate the `.png` files generated (e.g., `learning_curves_success.png`, `final_comparison.png`).
4.  Upload these images to your GitHub repository (usually in a folder named `assets` or `images`).
5.  Edit the `README.md` and replace the lines `> [Place your image here]` with standard Markdown image syntax:
    `![Description of Image](path/to/your/image.png)`

```
