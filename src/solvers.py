"""
Standard efficient dynamic-programming solvers for MDPs and ZSTBMGs.
"""
import numpy as np
from typing import Tuple

from src.models import MDP, ZSTBMG

def solve_mdp(mdp: MDP) -> Tuple[np.ndarray, np.ndarray]:
    """
    Solves a Finite-Horizon MDP using Backward Induction.

    Args:
        mdp: An MDP instance.
    
    Returns:
        values: optimal value function.
        policy: optimal deterministic policy.
    """
    H = mdp.horizon
    S = mdp.states_size
    
    # Base Case
    V = np.zeros((H + 1, S))
    policy = np.zeros((H, S), dtype=int)
    
    # Backward Induction
    for h in range(H - 1, -1, -1):
        # Q_h(s, a) = R_h(s, a) + sum_{s'} P_h(s'|s,a) * V_{h+1}(s')
        # We Transpose since shape of P is (A,S,S)
        Q_h = mdp.rewards[h].T + mdp.transitions[h] @ V[h+1]
        
        # V^*_h(s) = max_a Q_h(s,a)
        V[h] = np.max(Q_h, axis=0)
        policy[h] = np.argmax(Q_h, axis=0)
        
    return V, policy

def solve_zstbmg(game: ZSTBMG) -> Tuple[np.ndarray, np.ndarray]:
    """
    Solves a finite-horizon Zero-Sum Turn-Based Markov Game.
    
    Args:
        game: ZSTBMG instance.
              
    Returns:
        values: (H+1, S) value function for Player 1.
        policy: (H, S) optimal policy.
    """
    H = game.horizon
    S = game.states_size
    
    # V stores the value for Player 1
    V = np.zeros((H + 1, S))
    policy = np.zeros((H, S), dtype=int)
    
    # Mututally Recursive Backward Induction
    for h in range(H - 1, -1, -1):
        # 1. Compute Q-values for Player 1
        Q = game.rewards[h].T + game.transitions[h] @ V[h+1]
        
        # 2. Optimize based on whose turn it is (P1 Max/P2 Min)
        turn_mask = game.player_turn[h]

        # P1
        val_max = np.max(Q, axis=0)
        arg_max = np.argmax(Q, axis=0)
        
        # P2
        val_min = np.min(Q, axis=0)
        arg_min = np.argmin(Q, axis=0)
        
        # Select based on turn
        V[h] = np.where(game.player_turn[h] == 0, val_max, val_min)
        policy[h] = np.where(game.player_turn[h] == 0, arg_max, arg_min)

    return V, policy