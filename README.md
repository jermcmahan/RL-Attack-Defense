# Optimal Attack and Defense for Reinforcement Learning

The companion code to the AAAI paper [Optimal Attacks and Defense for Reinforcement Learning](https://ojs.aaai.org/index.php/AAAI/article/view/29346). Provides a framework for computing and simulating optimal adversarial attacks and optimal defense policies in Reinforcement Learning environments. In particular, our framework permits all possible attack surfaces: State, Perception, Action, and Reward. We visualize the impacts of these attacks, as well as the effectivenss of our robust defense policies, on a simple mini-grid environment.

## 🌟 Features

* **Multi-Surface Attack Framework:** An efficiently computable optimal attack framework that includes each online attack surface (test-time attack formulation).
* **Optimal Defense:** A game-theoretic defense mechanism where the agent learns a policy that is robust to the worst-case attacks (Minimax formulation).
* **Custom GridWorld Environment:** A flexible maze environment supporting custom layouts, hazards (lava), and goals to visualize our attacks and defenses.

## 📂 Project Structure

```
RL-Attack-Defense/
├── data/                   # Generated mazes and constraint files
├── results/                # Output plots and logs
├── scripts/
│   ├── generate_data.py    # Generates mazes and constraint masks
│   └── run_experiments.py  # Visualizes the attacks and defense
├── src/
│   ├── envs/
│   │   └── maze.py         # GridWorld environment
│   ├── models.py           # Data structures (MDP, Game)
│   ├── solvers.py          # MDP & Game solvers
│   ├── attack.py           # Optimal Attack Computation
│   ├── defense.py          # Optimal Defense Computation
│   └── simulation.py       # Attack Interaction Simulation
└── README.md
```

## 🚀 Quick Start

### 1. Installation
Clone the repository and install dependencies.

```
Bash
git clone https://github.com/jermcmahan/RL-Attack-Defense.git
cd RL-Attack-Defense
pip install -r requirements.txt
```

### 2. Data Generation

First, generate the maze layout and the constraint masks that define the "Danger Zones" (where the attacker has power).

```
Bash
# Generate the standard paper experiment data (Maze + Constraints)
python scripts/generate_data.py

# OR Generate a random maze
python scripts/generate_data.py --random --n 15 --p 0.2 --name my_random_maze
```

### 3. Run Experiments

Run the end-to-end experiment pipeline. This calculates the optimal baseline, computes the optimal strategy for all attacks, and solves for the robust action-defense policy.

```
Bash
python scripts/run_experiments.py
```

### 4. Visualizing Results

The results/ folder will contain visualizations of the agent's trajectories under attack:

* baseline.png: The optimal path with no interference.

* state_attack.png: The path taken when the agent is teleported.

* perceived_state_attack.png: The path taken when the agent is hallucinating.

* action_attack.png: The path taken when actions are overridden.

* robust_defense.png: The path of the robust agent surviving the action attack.

## 📊 Reproducibility
To reproduce the exact charts found in the report:

1. Run the full generation pipeline

2. Run the experiment suite

## 📜 Citation
If you use this code for your research, please cite:

```
Jeremy McMahan. (2025). Optimal Attack and Defense for Reinforcement Learning. 
GitHub Repository: https://github.com/jermcmahan/RL-Attack-Defense
```
