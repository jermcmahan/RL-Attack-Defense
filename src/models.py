"""
This module defines class representations of MDPs and ZSTBMGs
"""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MDP:
    """
    Non-Stationary Finite-Horizon Markov Decision Process.

    Attributes:
        horizon: The finite time horizon.
        rewards: The reward function (deterministic).
        transitions: The transition distribution.
        start: The start state.
    """
    horizon: int
    rewards: np.ndarray      # Shape: (H, S, A)
    transitions: np.ndarray  # Shape: (H, A, S, S)
    start: int

    @property
    def states_size(self) -> int:
        """Returns |S|"""
        return self.rewards.shape[1]

    @property
    def actions_size(self) -> int:
        """Returns |A|"""
        return self.rewards.shape[2]

    def __post_init__(self):
        H, S, A = self.rewards.shape
        
        if self.horizon != H:
            raise ValueError(f"Horizon mismatch: {self.horizon} vs shape {H}")
        
        if self.transitions.shape != (H, A, S, S):
             raise ValueError(f'Shape Mismatch: Transitions {self.transitions.shape} expected {(H, A, S, S)}')

        # Check probability sums for every timestep
        if not np.allclose(self.transitions.sum(axis=3), 1.0):
            raise ValueError('Invalid Transitions: probabilities do not sum to 1.0')


@dataclass(frozen=True)
class ZSTBMG:
    """
    Non-Stationary Finite-Horizon Zero-Sum Turn-Based Markov Game.
    
    Attributes:
        horizon: The finite time horizon
        rewards: The reward function for Player 1 (Maximizer).
        transitions: The transition distribution.
        player_turn: Indicator for whose turn it is.
        start: The start state.
    """
    horizon: int
    rewards: np.ndarray      # Shape: (H, S, A)
    transitions: np.ndarray  # Shape: (H, A, S, S)
    player_turn: np.ndarray  # Shape: (H, S)
    start: int

    @property
    def states_size(self) -> int:
        """Returns |S|"""
        return self.rewards.shape[1]

    @property
    def actions_size(self) -> int:
        """Returns |A|"""
        return self.rewards.shape[2]

    def __post_init__(self):
        H, S, A = self.rewards.shape
        
        if self.horizon != H:
            raise ValueError(f"Horizon mismatch: {self.horizon} vs shape {H}")
        
        if self.transitions.shape != (H, A, S, S):
             raise ValueError(f'Shape Mismatch: Transitions {self.transitions.shape} expected {(H, A, S, S)}')

        # Check probability sums for every timestep
        if not np.allclose(self.transitions.sum(axis=3), 1.0):
            raise ValueError('Invalid Transitions: probabilities do not sum to 1.0')
