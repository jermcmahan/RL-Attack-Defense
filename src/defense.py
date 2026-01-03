"""
Computes an optimal defense policy against action attacks.
"""
from typing import Tuple

import numpy as np

from src.models import MDP, ZSTBMG
from src.solvers import solve_zstbmg


def compute_defense(base_mdp: MDP, action_mask: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    Solves the Defense Game and extracts the Victim's policy.
    
    Args:
        base_mdp: Victim's MDP.
        action_mask: Defines which action attacks are allowed.
        
    Returns:
        V_victim: Victim's defense value. 
        pi_victim: Optimal defense policy.
    """
    defense_game = construct_defense_game(base_mdp, action_mask)
    
    # 1. Run the Generic Solver (Returns 2H horizon)
    V, pi = solve_zstbmg(defense_game)
    
    # 2. Extract Victim's Policy
    # Slice [::2] to take every even row and [:S] to ignore (s,a) states
    pi_victim = pi[::2, :base_mdp.states_size]
    
    # 3. Extract Value Function
    V_victim = V[::2, :base_mdp.states_size]

    return V_victim[0, base_mdp.start], pi_victim

def construct_defense_game(base_mdp: MDP, action_mask: np.ndarray) -> ZSTBMG:
    """
    Converts a Victim-Attacker-Interaction into a standard Zero-Sum Turn-Based Markov Game.
    
    Args:
        base_mdp: Victim's MDP.
        action_mask: Defines which action attacks are allowed.
        
    Returns:
        ZSTBMG: model of the victim-attacker interaction
    """
    H = base_mdp.horizon
    S = base_mdp.states_size
    A = base_mdp.actions_size
    
    # --- State Space Mapping ---
    # Flatten (s, a) -> S + s*A + a
    # 0 .. S-1              : Victim's States
    # S .. S + S*A - 1      : Attacker's States
    S_aug = S + S * A
    
    # New Horizon
    H_game = 2 * H
    
    # Initialize Arrays
    # Default reward is 0. transitions are 0.
    rewards = np.zeros((H_game, S_aug, A))
    transitions = np.zeros((H_game, A, S_aug, S_aug))
    player_turn = np.zeros((H_game, S_aug), dtype=int)

    for h in range(H):
        t_vic = 2 * h      # Victim's Turn
        t_att = 2 * h + 1  # Attacker's Turn
        
        # --- 1. Victim Step ---
        player_turn[t_vic, :S] = 0

        # Transitions: Choosing 'a' moves s -> (s,a)
        for s in range(S):
            for a in range(A):
                next_state = S + s * A + a
                transitions[t_vic, a, s, next_state] = 1.0
                # Reward is 0

        # Unreachable states self-loop for correctness
        for s_aug in range(S, S_aug):
            transitions[t_vic, :, s_aug, s_aug] = 1.0
                
        # --- 2. Attacker Step ---
        player_turn[t_att, S:S+S*A] = 1
        
        # Reachable states 
        for s in range(S):
            for a in range(A):
                curr_state = S + s * A + a
                
                # Attacker chooses an attack
                for attack in range(A):
                    if action_mask[h, s, a, attack]: 
                        # Valid Move
                        rewards[t_att, curr_state, attack] = base_mdp.rewards[h, s, attack]
                        transitions[t_att, attack, curr_state, :S] = base_mdp.transitions[h, attack, s, :]
                    else:
                        # Invalid Move for Attacker
                        rewards[t_att, curr_state, attack] = 1e9 # Penalty - helps victim
                        # Transition to self
                        transitions[t_att, attack, curr_state, curr_state] = 1.0

        # Unreachable states self-loop for correctness
        for s in range(S):
            transitions[t_att, :, s, s] = 1.0

    return ZSTBMG(H_game, rewards, transitions, player_turn, base_mdp.start)

