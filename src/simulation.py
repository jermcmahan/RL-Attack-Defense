"""
Methods for simulating agent-environment interactions under attack
"""
from dataclasses import dataclass, field
from typing import List

import numpy as np

from src.models import MDP

@dataclass
class Trajectory:
    states: List[int] = field(default_factory=list)
    actions: List[int] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)

def simulate_clean(base_mdp: MDP, pi: np.ndarray) -> Trajectory:
    """Simulates a policy in a clean environment with no attack."""
    trajectory = Trajectory()
    state = base_mdp.start

    for t in range(base_mdp.horizon):
        trajectory.states.append(state)
        probs = base_mdp.transitions[t, pi[t, state], state]
        next_state = np.random.choice(len(probs), p=probs)
        trajectory.rewards.append(base_mdp.rewards[t, state, pi[t, state]])
        state = next_state
    
    return trajectory

def simulate_attack(env: MDP, surface: str, victim_pi: np.ndarray, attacker_pi: np.ndarray) -> Trajectory:
    """Master Simulator"""
    match surface:
        case 'state':
            return run_state_attack_sim(env, victim_pi, attacker_pi)
        case 'perceived-state':
            return run_perceived_attack_sim(env, victim_pi, attacker_pi)
        case 'action':
            return run_action_attack_sim(env, victim_pi, attacker_pi)
        case _:
            raise ValueError(f'Attack Surface ({surface}) is Unkown')

def run_state_attack_sim(env: MDP, victim_pi: np.ndarray, attacker_pi: np.ndarray) -> Trajectory:
    """
    Simulates a State Attack.
    
    Flow:
    1. Env is at s.
    2. Attacker chooses s_dag = nu(s).
    3. Victim observes s_dag, chooses a = pi(s_dag).
    4. Physics executes P(s' | s_dag, a).
    """
    trajectory = Trajectory()
    state = env.start
    
    for h in range(env.horizon):
        # 1. Attacker Action
        state_attack = attacker_pi[h, state]
        
        # 2. Victim Action
        action = victim_pi[h, state_attack]
        
        # 3. Dynamics
        probs = env.transitions[h, action, state_attack]
        next_state = np.random.choice(len(probs), p=probs)
        reward = env.rewards[h, state_attack, action]
        
        # Log
        trajectory.states.append(state)
        trajectory.actions.append(action)
        trajectory.rewards.append(reward)
        
        # Update
        state = next_state
        
    return trajectory

def run_perceived_attack_sim(env: MDP, victim_pi: np.ndarray, attacker_pi: np.ndarray) -> Trajectory:
    """
    Simulates a Perceived State Attack.
    
    Flow:
    1. Env is at s.
    2. Attacker chooses s_dag = nu(s).
    3. Victim observes s_dag, chooses a = pi(s_dag).
    4. Physics executes P(s' | s_true, a). 
    """
    trajectory = Trajectory()
    state = env.start
    
    for h in range(env.horizon):
        # 1. Attacker Action
        perceived_attack = attacker_pi[h, state]
        
        # 2. Victim Action
        action = victim_pi[h, perceived_attack]
        
        # 3. Dynamics
        probs = env.transitions[h, action, state]
        next_state = np.random.choice(len(probs), p=probs)
        reward = env.rewards[h, state, action]
        
        trajectory.states.append(state)
        trajectory.actions.append(action)
        trajectory.rewards.append(reward)
        
        state = next_state

    return trajectory

def run_action_attack_sim(env: MDP, victim_pi: np.ndarray, attacker_pi: np.ndarray) -> Trajectory:
    """
    Simulates an Action Attack.
    
    Flow:
    1. Env is at s.
    2. Attacker chooses a_dag = nu(s).
    3. Physics executes P(s' | s, a_dag).
    """
    trajectory = Trajectory()
    state = env.start
    
    for h in range(env.horizon):
        # 1. Attacker Action
        action_attack = attacker_pi[h, state]
        
        # 2. Dynamics
        probs = env.transitions[h, action_attack, state]
        next_state = np.random.choice(len(probs), p=probs)
        reward = env.rewards[h, state, action_attack]
        
        trajectory.states.append(state)
        trajectory.actions.append(action_attack)
        trajectory.rewards.append(reward)
        
        state = next_state
        
    return trajectory
