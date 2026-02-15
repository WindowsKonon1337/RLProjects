"""
Policy Definitions
Trainable policy over 4 directions (Left, Up, Right, Down)
using per-state logits and softmax probabilities.
"""

import numpy as np
from config import VERBOSE_CONSOLE

ACTIONS = ["Left", "Up", "Right", "Down"]

# Global trainable parameters:
# THETA[(x, y)] = np.array([logit_left, logit_up, logit_right, logit_down])
THETA = {}


def reset_theta():
    """
    Reset trainable policy parameters.

    Clears all stored logits for states, so policy starts from scratch.
    """
    global THETA
    THETA = {}


def get_theta():
    """
    Get current trainable policy parameters.

    Returns:
        dict: Mapping {(x, y): np.array shape (4,)} with action logits.
    """
    return THETA


def set_theta(theta):
    """
    Set policy parameters from external training code.

    Args:
        theta (dict): Mapping {(x, y): np.array shape (4,)}.
    """
    global THETA
    THETA = theta


def _softmax(x):
    """
    Compute softmax probabilities from logits.

    Args:
        x (np.array): 1D array of action logits.

    Returns:
        np.array: Probability distribution with sum = 1.
    """
    z = x - np.max(x)  # numerical stability
    e = np.exp(z)
    return e / (np.sum(e) + 1e-12)


def _valid_mask(s):
    """
    Build validity mask for 4 actions from state neighbors.

    Neighbors expected order:
    0) Current
    1) Left
    2) Up
    3) Right
    4) Down

    Args:
        s (dict): State dict with key 'neighbors'.

    Returns:
        np.array: Boolean mask of shape (4,), True for valid actions.
    """
    neighbors_4 = s["neighbors"][1:]  # [Left, Up, Right, Down]
    # neighbor tuple: (x, y, is_obstacle, is_visited)
    return np.array([not n[2] for n in neighbors_4], dtype=bool)


def _ensure_state_params(pos):
    """
    Ensure logits exist for a given position.

    Initializes zero logits for unseen state (x, y).

    Args:
        pos (tuple): Agent position (x, y).
    """
    if pos not in THETA:
        THETA[pos] = np.zeros(4, dtype=np.float64)


def policy(s):
    """
    Compute policy probabilities for current state.

    Uses per-state trainable logits and masks invalid actions
    (obstacles/outside grid).

    Args:
        s (dict): State dict.

    Returns:
            - np.array: Action probabilities of shape (4,)
    """
    pos = s["agent_pos"]
    _ensure_state_params(pos)

    logits = THETA[pos].copy()
    logits = np.clip(logits, -10.0, 10.0)
    mask = _valid_mask(s)

    # Set invalid actions to very negative logit
    logits[~mask] = -1e9
    probs = _softmax(logits)

    if VERBOSE_CONSOLE:
        print(f"  [policy] s={pos} (step {s['steps']}):")
        for i, p in enumerate(probs):
            print(f"    P({ACTIONS[i]}) = {p:.4f} (valid={mask[i]})")

    return probs
