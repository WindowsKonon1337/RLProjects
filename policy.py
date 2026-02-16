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

def state_key(s):
    """
    Build compact contextual key:
      (x, y, action_flags)

    action_flags for [Left, Up, Right, Down]:
      -1 = invalid (obstacle/outside)
       0 = valid but visited
       1 = valid and unvisited
    """
    x, y = s["agent_pos"]
    n4 = s["neighbors"][1:]  # Left, Up, Right, Down

    flags = []
    for n in n4:
        is_obstacle = bool(n[2])  # tuple: (x, y, is_obstacle, is_visited)
        is_visited = bool(n[3])

        if is_obstacle:
            flags.append(-1)
        elif is_visited:
            flags.append(0)
        else:
            flags.append(1)

    return (x, y, tuple(flags))


def _ensure_state_params(key):
    """
    Ensure logits exist for a given position.

    Initializes zero logits for unseen state (x, y).

    Args:
        pos (tuple): Agent position (x, y).
    """
    if key not in THETA:
        THETA[key] = np.zeros(4, dtype=np.float64)


def policy(s):
    """
    Compute trainable policy probabilities for 4 actions:
    [Left, Up, Right, Down].

    Uses contextual key (x, y, action_flags), masks invalid actions,
    and slightly prefers valid-unvisited moves.
    """
    key = state_key(s)
    _ensure_state_params(key)

    logits = THETA[key].copy()

    # action_flags: -1 invalid, 0 visited-valid, 1 unvisited-valid
    flags = key[2]

    # Small bias: prefer unvisited valid cells, but do NOT forbid visited
    for i, f in enumerate(flags):
        if f == 1:
            logits[i] += 0.25
        elif f == 0:
            logits[i] += 0.0
        else:  # invalid
            logits[i] = -1e9

    # Optional safety clip
    logits = np.clip(logits, -10.0, 10.0)

    probs = _softmax(logits)

    if VERBOSE_CONSOLE:
        print(f"  [policy] s={s['agent_pos']} key={key[2]} (step {s['steps']}):")
        for i, p in enumerate(probs):
            valid = (flags[i] != -1)
            print(f"    P({ACTIONS[i]}) = {p:.4f} (valid={valid})")

    return probs
