"""
A simple mini-grid environment for experiments.
"""
import os
import copy
from typing import List, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from src.models import MDP

class Maze:
    STAY = 0
    LEFT = 1
    RIGHT = 2
    UP = 3
    DOWN = 4
    
    ACTIONS = [STAY, LEFT, RIGHT, UP, DOWN]
    ACTION_NAMES = ["Stay", "Left", "Right", "Up", "Down"]

    def __init__(self, grid: np.ndarray, horizon: int):
        """
        Args:
            grid: Binary numpy array (N, N). 1 = Safe, 0 = Lava.
            horizon: Time horizon for the MDP.
        """
        self.grid = grid
        self.size = grid.shape[0]
        self.horizon = horizon
        self.start_pos = (0, 0)
        self.goal_pos = (self.size - 1, self.size - 1)
        
        # Validation
        if self.grid[self.start_pos] == 0:
            raise ValueError("Start position cannot be Lava.")
        if self.grid[self.goal_pos] == 0:
            raise ValueError("Goal position cannot be Lava.")

    @classmethod
    def generate(cls, n: int, p_lava: float, horizon: Optional[int] = None) -> 'Maze':
        """
        Generates a random N x N maze with p_percent lava.
        """
        # Calculate number of safe/lava cells
        n_cells = n * n
        n_lava = int(np.floor(n_cells * p_lava))
        n_safe = n_cells - n_lava
        
        # Create flat array
        layout = np.array([0] * n_lava + [1] * n_safe, dtype=np.int64)
        np.random.shuffle(layout)
        
        # Reshape to grid
        grid = layout.reshape((n, n))
        
        # Enforce Safe Start/Goal
        grid[0, 0] = 1
        grid[n-1, n-1] = 1
        
        if horizon is None:
            horizon = 2 * n # Default heuristic
            
        return cls(grid, horizon)

    def to_mdp(self) -> MDP:
        """
        Converts the static Grid into the MDP format.
        """
        N = self.size
        S = N * N
        A = len(self.ACTIONS)
        H = self.horizon
        P_step = np.zeros((A, S, S))
        R_step = np.zeros((S, A))
        
        for r in range(N):
            for c in range(N):
                s = self._coord_to_state(r, c)
                deltas = [(0,0), (0,-1), (0,1), (-1,0), (1,0)]
                
                for action, (dr, dc) in enumerate(deltas):
                    next_r, next_c = r + dr, c + dc
                    
                    # 1. Check Bounds (Wall Collision)
                    if not (0 <= next_r < N and 0 <= next_c < N):
                        # Hit Wall -> Stay in current state 's'
                        next_s = s
                    else:
                        next_s = self._coord_to_state(next_r, next_c)
                    
                    # 2. Set Transition
                    P_step[action, s, next_s] = 1.0
                    
                    # 3. Set Reward
                    # A. Goal Rewards
                    if (next_r, next_c) == self.goal_pos:
                        R_step[s, action] = 1.0
                    
                    # B. Lava Penalties (Overwrites Goal reward if Goal was lava
                    if 0 <= next_r < N and 0 <= next_c < N: # If valid cell
                        if self.grid[next_r, next_c] == 0: # It is Lava
                            R_step[s, action] = -float(H)
        
        # Broadcast to Horizon H
        transitions = np.tile(P_step[np.newaxis, :, :, :], (H, 1, 1, 1))
        
        # Rewards: (H, S, A)
        rewards = np.tile(R_step[np.newaxis, :, :], (H, 1, 1))
        
        return MDP(H, rewards, transitions, start=self._coord_to_state(0,0))

    def visualize(self, trajectory: List[int] = None, save_path: str = None, title: str = "Maze"):
        """
        Visualizes the maze and an optional path with context-aware markers.
        """
        # Create a display grid: 0=Lava, 1=Safe
        display_grid = self.grid.copy().astype(float)
        
        # Plotting Setup
        fig, ax = plt.subplots(figsize=(6, 6))
        
        # Custom Colormap
        # 0.0 (Lava) -> Dark Red / Black
        # 1.0 (Safe) -> White / Light Grey
        cmap = colors.ListedColormap(['#4a0404', '#f0f0f0']) 
        bounds = [0, 0.5, 1.5]
        norm = colors.BoundaryNorm(bounds, cmap.N)
        
        ax.imshow(self.grid, cmap=cmap, norm=norm, origin='upper')
        
        # Draw Grid lines
        ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=1, alpha=0.1)
        ax.set_xticks(np.arange(-.5, self.size, 1))
        ax.set_yticks(np.arange(-.5, self.size, 1))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        
        # Mark Start/Goal Text
        ax.text(self.start_pos[1], self.start_pos[0], 'S', ha='center', va='center', 
                color='green', fontweight='bold', fontsize=14)
        ax.text(self.goal_pos[1], self.goal_pos[0], 'G', ha='center', va='center', 
                color='gold', fontweight='bold', fontsize=14)
        
        # Plot Path
        if trajectory:
            path_coords = [self._state_to_coord(s) for s in trajectory]
            
            # Separate X and Y for plotting
            py = [c[0] for c in path_coords] # Rows (Y)
            px = [c[1] for c in path_coords] # Cols (X)
            
            # Plot the line
            ax.plot(px, py, color='royalblue', linewidth=3, alpha=0.7, label='Trajectory')
            
            # --- SMART MARKERS ---
            
            # 1. Start Marker
            ax.plot(px[0], py[0], color='green', marker='o', markersize=8, zorder=10)
            
            # 2. End Marker (Context Sensitive)
            end_r, end_c = py[-1], px[-1]
            end_state_val = self.grid[end_r, end_c]
            
            if (end_r, end_c) == self.goal_pos:
                # SUCCESS: Gold Star
                ax.plot(end_c, end_r, color='gold', marker='*', markersize=18, 
                        markeredgecolor='black', markeredgewidth=1.5, zorder=10, label='Goal Reached')
            elif end_state_val == 0:
                # FAILURE: Red X (In Lava)
                ax.plot(end_c, end_r, color='red', marker='X', markersize=12, 
                        markeredgecolor='white', markeredgewidth=1, zorder=10, label='Terminated (Lava)')
            else:
                # TIMEOUT: Blue Square (Ended in safe space)
                ax.plot(end_c, end_r, color='blue', marker='s', markersize=10, 
                        markeredgecolor='white', zorder=10, label='Timeout')

        ax.set_title(title, fontsize=14, pad=10)
        
        # Legend
        # ax.legend(loc='upper right')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def _coord_to_state(self, r: int, c: int) -> int:
        return r * self.size + c
    
    def _state_to_coord(self, s: int) -> Tuple[int, int]:
        return divmod(s, self.size)
        
    def get_distance_mask(self, dist: int) -> np.ndarray:
        """
        Helper for Attacks: Returns boolean mask (S, S) where mask[s, s'] is True 
        if Manhattan distance between s and s' <= dist.
        """
        S = self.size * self.size
        mask = np.zeros((S, S), dtype=bool)
        
        for s1 in range(S):
            r1, c1 = self._state_to_coord(s1)
            for s2 in range(S):
                r2, c2 = self._state_to_coord(s2)
                d = abs(r1 - r2) + abs(c1 - c2)
                if d <= dist:
                    mask[s1, s2] = True
        return mask

    def save(self, file_path: str):
        """Saves the maze grid to a .npy file."""
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        np.save(file_path, self.grid)

    @classmethod
    def load(cls, file_path: str, horizon: Optional[int] = None) -> 'Maze':
        """
        Loads a maze grid from a .npy file.
        
        Args:
            file_path: Path to the .npy file.
            horizon: Optional horizon override. If None, defaults to 2 * N.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"No maze found at {file_path}")
            
        grid = np.load(file_path)
        
        if horizon is None:
            horizon = 2 * grid.shape[0]
            
        return cls(grid, horizon)
