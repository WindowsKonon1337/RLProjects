"""
Policy Definitions
Contains the agent's policy function.
"""

import numpy as np
from config import VERBOSE_CONSOLE


def policy(s):
    """
    Random policy over 4 directions (neighbors)
    
    Args:
        s: State dict
        
    Returns:
        np.array: Probability distribution over actions
    """
    neighbors = s['neighbors']

    num_actions = len(neighbors) - 1
    non_norm = np.random.random_sample(num_actions) 
    probs = non_norm / non_norm.sum()
    
    if VERBOSE_CONSOLE:
        print(f"  [policy] s={s['agent_pos']} (step {s['steps']}):")
        directions = ["Left", "Up", "Right", "Down"]
        for i, p in enumerate(probs):
            print(f"    P({directions[i]}) = {p:.4f}")
            
    return probs
