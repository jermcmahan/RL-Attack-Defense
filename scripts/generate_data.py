"""
Generates maze datasets and corresponding attack constraints.
Supports both the fixed "Paper" experiment and random maze generation.
"""

import argparse
import os
import numpy as np
from datetime import datetime
from src.envs.maze import Maze
from src.attack import AttackConstraints

# =============================================================================
# Helper Functions: Constraint Mask Generation
# =============================================================================

def create_state_mask(mdp, danger_mask: np.ndarray) -> np.ndarray:
    """
    Creates a State Attack mask (H, S, S).
    - Default: Identity (Target = Self)
    - Danger Zone: Can attack ANY target state.
    """
    H, S = mdp.horizon, mdp.states_size
    
    # 1. Default: Identity
    mask = np.eye(S, dtype=bool)
    mask = np.tile(mask[np.newaxis, :, :], (H, 1, 1))
    
    # 2. Danger Zone: Allow all targets
    # mask[h, s, :] = True means if current state is s, we can teleport anywhere
    for h in range(H):
        mask[h, danger_mask, :] = True
        
    return mask

def create_perc_mask(mdp, danger_mask: np.ndarray) -> np.ndarray:
    """
    Creates a Perception Attack mask (H, S, S).
    - Default: Identity (Perc = Phys)
    - Danger Zone: Can cause hallucination of ANY state.
    """
    # Logic is identical to state mask (Identity default, full row allowed in danger)
    return create_state_mask(mdp, danger_mask)

def create_action_mask(mdp, danger_mask: np.ndarray) -> np.ndarray:
    """
    Creates an Action Attack mask (H, S, A, A).
    - Default: Identity (Forced = Intended)
    - Danger Zone: Can force ANY action.
    """
    H, S, A = mdp.horizon, mdp.states_size, mdp.actions_size
    
    # 1. Default: Identity (Forced == Intended)
    mask = np.zeros((H, S, A, A), dtype=bool)
    for a in range(A):
        mask[:, :, a, a] = True
        
    # 2. Danger Zone: Allow forcing ANY action
    # mask[h, s, :, :] = True means if current state is s, any intended action can become any forced action
    for h in range(H):
        mask[h, danger_mask, :, :] = True
        
    return mask

# =============================================================================
# 1. Paper Experiment Data
# =============================================================================

def generate_paper_data(output_dir: str):
    """Generates the specific Maze and Constraints from the paper."""
    print(f"Generating Paper Experiment Data in {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)
    
    # --- 1. Maze ---
    grid_layout = np.array([
        [1, 1, 1, 1, 1, 1, 0, 1, 0, 1],
        [1, 1, 0, 1, 1, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 1, 0, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 1, 1, 1, 0, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 0, 0, 1, 1, 0, 1, 1, 1],
        [0, 1, 1, 1, 0, 1, 0, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 0, 1, 1, 1]
    ])
    maze = Maze(grid_layout, horizon=20)
    maze.save(os.path.join(output_dir, "paper_maze.npy"))
    mdp = maze.to_mdp()

    # --- 2. Define Danger Zones ---
    # Zone A: Top corridor (Rows 0-1, Cols 7+)
    danger_zone_A = np.zeros(mdp.states_size, dtype=bool)
    for r in range(0, 2):
        for c in range(7, maze.size):
            danger_zone_A[maze._coord_to_state(r, c)] = True

    # Zone B: Top-Left area (Rows 0-4, Cols 5+)
    danger_zone_B = np.zeros(mdp.states_size, dtype=bool)
    for r in range(0, 5):
        for c in range(5, maze.size):
            danger_zone_B[maze._coord_to_state(r, c)] = True

    # --- 3. Generate Constraints ---
    
    # State Attack (Uses Zone A)
    s_mask = create_state_mask(mdp, danger_zone_A)
    AttackConstraints(s_mask).save(os.path.join(output_dir, "state_constraints_paper.npz"))

    # Perceived Attack (Uses Zone A)
    p_mask = create_perc_mask(mdp, danger_zone_A)
    AttackConstraints(p_mask).save(os.path.join(output_dir, "perceived_constraints_paper.npz"))

    # Action Attack (Uses Zone B - Larger)
    a_mask = create_action_mask(mdp, danger_zone_B)
    AttackConstraints(a_mask).save(os.path.join(output_dir, "action_constraints_paper.npz"))
    
    print("Done.")

# =============================================================================
# 2. Random Experiment Data
# =============================================================================

def generate_random_data(n: int, p_lava: float, output_dir: str, name: str = None):
    """Generates random maze and constraints."""
    os.makedirs(output_dir, exist_ok=True)
    if name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"random_{n}x{n}_{timestamp}"
    
    print(f"Generating Random Data '{name}' in {output_dir}...")

    # 1. Maze
    maze = Maze.generate(n=n, p_lava=p_lava)
    maze_filename = f"{name}_maze.npy"
    maze.save(os.path.join(output_dir, maze_filename))
    mdp = maze.to_mdp()

    # 2. Define Danger Zone (Random 20% of states)
    n_states = mdp.states_size
    n_danger = int(0.2 * n_states)
    danger_indices = np.random.choice(n_states, n_danger, replace=False)
    
    danger_mask = np.zeros(n_states, dtype=bool)
    danger_mask[danger_indices] = True
    
    # 3. Generate Constraints (Using same random zone for all attacks for simplicity)
    s_mask = create_state_mask(mdp, danger_mask)
    AttackConstraints(s_mask).save(os.path.join(output_dir, f"state_constraints_{name}.npz"))

    p_mask = create_perc_mask(mdp, danger_mask)
    AttackConstraints(p_mask).save(os.path.join(output_dir, f"perceived_constraints_{name}.npz"))

    a_mask = create_action_mask(mdp, danger_mask)
    AttackConstraints(a_mask).save(os.path.join(output_dir, f"action_constraints_{name}.npz"))
    
    print(f"  - Saved all files with prefix: {name}")

# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Maze and Attack Constraint Data")
    
    # Mode selection
    parser.add_argument("--random", action="store_true", help="Generate random data instead of paper data")
    
    # Random generation arguments
    parser.add_argument("--n", type=int, default=10, help="Grid size (N x N) for random maze")
    parser.add_argument("--p", type=float, default=0.2, help="Lava probability for random maze")
    parser.add_argument("--name", type=str, default=None, help="Custom name for random run")
    
    # General arguments
    parser.add_argument("--output_dir", type=str, default="data", help="Folder to save data")
    
    args = parser.parse_args()
    
    if args.random:
        generate_random_data(args.n, args.p, args.output_dir, args.name)
    else:
        generate_paper_data(args.output_dir)
