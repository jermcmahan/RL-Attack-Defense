"""
Optimal Attack Computation
"""
from typing import Tuple
from dataclasses import dataclass
import os

import numpy as np

from src.models import MDP
from src.solvers import solve_mdp

@dataclass
class AttackConstraints:
    """
    Container for loading a constraint mask from disk.
    """
    mask: np.ndarray

    def save(self, file_path: str):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        np.savez_compressed(file_path, mask=self.mask)

    @classmethod
    def load(cls, file_path: str) -> 'AttackConstraints':
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Constraint file not found: {file_path}")
        data = np.load(file_path)
        return cls(data['mask'])


def compute_attack(base_mdp: MDP, policy: np.ndarray, surface: Tuple[str, np.ndarray]) -> Tuple[float, np.ndarray]:
    """
    Computes the optimal attack for the attacker on the given surfaces.

    Args:
        base_mdp: The victim's true environment.
        policy: The victim's fixed policy.
        surfaces: The attack surface and corresponding available attacks mask.

    Returns:
        Optimal Attack Policy and Value
    """

    surface_name, allowed_attacks = surface

    match surface_name:
        case 'state':
            attacker_mdp = construct_state_attack_mdp(base_mdp, policy, allowed_attacks)
        case 'perceived-state':
            attacker_mdp = construct_perceived_attack_mdp(base_mdp, policy, allowed_attacks)
        case 'action':
            attacker_mdp = construct_action_attack_mdp(base_mdp, policy, allowed_attacks)
        case _:
            raise ValueError(f'Attack Surface ({surface_name}) Unknown')

    val, attack = solve_mdp(attacker_mdp)
    return val[0,attacker_mdp.start], attack



def construct_state_attack_mdp(base_mdp: MDP, policy: np.ndarray, state_mask: np.ndarray) -> MDP:
    """
    Constructs the State Attack MDP.
    
    Args:
        base_mdp: Victim MDP (H, A, S, S).
        policy: Victim policy (H, S).
        state_mask: Boolean mask (H, S, S_dag).

    Returns:
        MDP: The MDP representing the attacker's problem.
    """
    H = base_mdp.horizon
    S = base_mdp.states_size
    
    # 1. Attacker Transitions
    attacker_transitions = np.zeros((H, S, S, S))
    
    for h in range(H):
        for s_dag in range(S):
            # P_dag(s' | s, s_dag) = P(s' | s_dag, pi_h(s_dag))
            attacker_transitions[h,s_dag,:,:] = base_mdp.transitions[h,policy[h,s_dag],s_dag,:]

    # 2. Attacker Rewards
    attacker_rewards = np.zeros((H, S, S))
    
    for h in range(H):
        for s_dag in range(S):
            # r_dag(s,s_dag) = -r(s_dag, pi[h,s_dag])
            attacker_rewards[h,:,s_dag] = -base_mdp.rewards[h,s_dag,policy[h,s_dag]]
    
    attacker_rewards[~state_mask] = -1e9

    return MDP(H, attacker_rewards, attacker_transitions, base_mdp.start)

def construct_perceived_attack_mdp(base_mdp: MDP, policy: np.ndarray, perceived_mask: np.ndarray) -> MDP:
    """
    Constructs the Perceived-State Attack MDP.

    Args:
        base_mdp: Victim MDP (H, A, S, S).
        policy: Victim policy (H, S).
        perceived_mask: Boolean mask (H, S, S_dag).

    Returns:
        MDP: The MDP representing the attacker's problem.
    """
    H = base_mdp.horizon
    S = base_mdp.states_size

    # 1. Attacker Transitions
    attacker_transitions = np.zeros((H, S, S, S))

    for h in range(H):
        for s_dag in range(S):
            # P_dag(s' | s, s_dag) = P(s' | s, pi_h(s_dag))
            attacker_transitions[h,s_dag,:,:] = base_mdp.transitions[h,policy[h,s_dag],:,:]

    # 2. Attacker Rewards
    attacker_rewards = np.zeros((H, S, S))

    for h in range(H):
        for s_dag in range(S):
            # r_dag(s,s_dag) = -r(s, pi[h,s_dag])
            attacker_rewards[h,:,s_dag] = -base_mdp.rewards[h, :, policy[h,s_dag]]
    
    attacker_rewards[~perceived_mask] = -1e9

    return MDP(H, attacker_rewards, attacker_transitions, base_mdp.start)

def construct_action_attack_mdp(base_mdp: MDP, policy: np.ndarray, action_mask: np.ndarray) -> MDP:
    """
    Constructs the Action Attack MDP.

    Args:
        base_mdp: Victim MDP (H, A, S, S).
        policy: Victim policy (H, S).
        action_mask: Boolean mask (H, S, A, A_dag).

    Returns:
        MDP: The MDP representing the attacker's problem.
    """
    H, S, A = base_mdp.horizon, base_mdp.states_size, base_mdp.actions_size
    
    # 1. Attacker Transitions
    attacker_transitions = np.zeros((H, A, S, S))

    for h in range(H):
        for a_dag in range(A):
            # P_dag(s' | s, a_dag) = P(s' | s, a_dag)
            attacker_transitions[h,a_dag,:,:] = base_mdp.transitions[h,a_dag,:,:]

    # 2. Attacker Rewards
    attacker_rewards = np.zeros((H, S, A))

    for h in range(H):
        for a_dag in range(A):
            # r_dag(s,a_dag) = -r(s,a_dag)
            attacker_rewards[h,:,a_dag] = -base_mdp.rewards[h,:,a_dag]
    
    attacker_rewards[~action_mask[np.arange(H)[:,None],np.arange(S)[None,:],policy,:]] = -1e9

    return MDP(H, attacker_rewards, attacker_transitions, base_mdp.start)
