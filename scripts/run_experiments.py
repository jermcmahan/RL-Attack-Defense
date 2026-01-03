"""
End-to-end experiment runner for "Optimal Attack and Defense for RL".
Computes and simulates attacks/defenses.
"""

import os
import argparse
import numpy as np
from typing import Dict, Any

from src.models import MDP
from src.envs.maze import Maze
from src.solvers import solve_mdp
from src.attack import compute_attack, AttackConstraints
from src.defense import compute_defense
from src.simulation import simulate_attack, simulate_clean, Trajectory

# =============================================================================
# Configuration
# =============================================================================

# This dictionary maps the display name to the parameters needed for the master methods.
ATTACK_CONFIGS = {
    "State Attack": {
        "type": "state",  # String passed to compute/simulate
        "filename": "state_constraints_paper.npz",
        "title": "State Attack"
    },
    "Perceived-State Attack": {
        "type": "perceived-state",
        "filename": "perceived_constraints_paper.npz",
        "title": "Perceived-State Attack"
    },
    "Action Attack": {
        "type": "action",
        "filename": "action_constraints_paper.npz",
        "title": "Action Attack"
    }
}

# =============================================================================
# Main Loop
# =============================================================================

def main(maze_path: str, data_dir: str, output_dir: str):
    print(f"=== Initialize Experiment ===")
    
    # 1. Load Maze
    if not os.path.exists(maze_path):
        raise FileNotFoundError(f"Maze not found at: {maze_path}")
        
    temp_maze = Maze.load(maze_path)
    # Heuristic: 20 steps for small mazes, 2*N for large ones
    h = 20 if temp_maze.size == 10 else 2 * temp_maze.size
    maze = Maze.load(maze_path, horizon=h)
    base_mdp = maze.to_mdp()
    print(f"  - Loaded Maze: {maze.size}x{maze.size}, Horizon: {maze.horizon}")

    # Setup Output
    maze_name = os.path.splitext(os.path.basename(maze_path))[0]
    run_output = os.path.join(output_dir, maze_name)
    os.makedirs(run_output, exist_ok=True)
    print(f"  - Saving results to: {run_output}/")

    # 2. Run Baseline --------------------------------------------------------
    print("\n=== Computing Optimal Baseline Policy ===")
    
    # 1. Solve
    _, pi_star = solve_mdp(base_mdp)
    traj = simulate_clean(base_mdp, pi_star)
    print(f"  - Baseline Reward: {sum(traj.rewards):.2f}")
    maze.visualize(traj.states, save_path=f"{run_output}/baseline.png", title="Optimal Policy (No Attack)")

    # 3. Run Attacks ---------------------------------------------------------
    print("\n=== Computing Attacks ===")
    
    for name, config in ATTACK_CONFIGS.items():
        print(f"\n--- {name} ---")
        
        # A. Load Constraints
        c_path = os.path.join(data_dir, config["filename"])
        if not os.path.exists(c_path):
            print(f"  ! Skipped (File not found: {config['filename']})")
            continue
            
        constraints = AttackConstraints.load(c_path)
        
        # B. Compute Attack
        print(f"  - Computing Optimal Attack Strategy...")
        _, nu_attack = compute_attack(base_mdp, pi_star, (config["type"], constraints.mask))
        
        # C. Simulate
        print(f"  - Simulating...")
        traj = simulate_attack(base_mdp, config["type"], pi_star, nu_attack)
        
        print(f"  - Reward: {sum(traj.rewards):.2f}")
        
        # D. Visualize
        file_suffix = config["type"].replace("-", "_") + '_attack'
        maze.visualize(traj.states, save_path=f"{run_output}/{file_suffix}.png", title=config["title"])

    # 4. Run Defense -----------------------------------------------------------
    print("\n=== Computing Defense (Action) ===")
    
    act_c_path = os.path.join(data_dir, "action_constraints_paper.npz")
    if os.path.exists(act_c_path):
        constraints = AttackConstraints.load(act_c_path)
        
        # A. Compute Robust Policy
        print("  - Computing Robust Policy (Minimax)...")
        V_rob, pi_rob = compute_defense(base_mdp, constraints.mask)
        
        # B. Attack the Robust Policy
        print("  - Computing Best Response to Defense...")
        _, nu_rob = compute_attack(base_mdp, pi_rob, ("action", constraints.mask))
        
        # C. Simulate
        traj_rob = simulate_attack(base_mdp, "action", pi_rob, nu_rob)
        
        print(f"  - Reward: {sum(traj_rob.rewards):.2f}")
        maze.visualize(traj_rob.states, save_path=f"{run_output}/robust_defense.png", title="Robust Defense")
        
    else:
        print("  ! Skipped Defense (Action constraints not found)")

    print("\n=== Experiments Complete ===")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RL Attack/Defense Experiments")
    
    parser.add_argument("--maze_path", type=str, default="data/paper_maze.npy", 
                        help="Path to the maze file.")
    parser.add_argument("--data_dir", type=str, default="data", 
                        help="Directory containing constraint .npz files.")
    parser.add_argument("--output_dir", type=str, default="results", 
                        help="Directory to save results.")
    
    args = parser.parse_args()
    main(args.maze_path, args.data_dir, args.output_dir)
