"""
REINFORCE update for episodic trajectories.

Implements the policy-gradient estimator in the form:
∇θ J = E_θ [ G(τ) * Σ_t ∇θ log π_θ(a_t | s_t) ]

Here:
- Policy parameters are stored in policy.THETA as:
    THETA[(x, y)] = np.array([logit_left, logit_up, logit_right, logit_down])
- A trajectory is expected as a list of dicts:
    {
        "state":  s_t,              # state BEFORE action
        "action": a_t,              # int: 0..3
        "reward": r_t,              # reward after transition
        "probs":  pi(.|s_t)         # np.array shape (4,), used for sampling
    }
"""

# reinforce.py
import numpy as np
from collections import deque
from policy import get_theta, set_theta, state_key
from rl_functions import sample_step, reward

BASELINE_V = {}
G_HISTORY = deque(maxlen=100)

def collect_trajectory(graph, init_state, max_steps):
    """
    Collect one episode trajectory.

    Args:
        graph: Environment graph.
        init_state (dict): Initial state.
        max_steps (int): Max steps per episode.

    Returns:
        list: Trajectory with items:
              {"state", "action", "reward", "probs"}
    """
    traj = []
    s = init_state

    for _ in range(max_steps):
        # sample_step must return: (next_state, action_idx, probs)
        s_next, a, probs = sample_step(graph, s)
        r = reward(graph, s_next)

        traj.append({
            "state": s,       # state BEFORE action
            "action": a,      # int 0..3
            "reward": r,      # reward after transition
            "probs": probs    # pi(.|s) used for sampling
        })

        s = s_next

    return traj


def _episode_return(traj, gamma=1.0):
    """
    Compute discounted episode return G(tau).
    """
    G = 0.0
    coef = 1.0
    for step in traj:
        G += coef * float(step["reward"])
        coef *= gamma
    return G


def reinforce_update(traj, alpha=0.001, gamma=1.0, beta_v=0.2, use_state_baseline=True):
    """
    One REINFORCE update with formula:

      grad = sum_t (G - b_t) * grad log pi(a_t|s_t)
      theta <- theta + alpha * grad
      - b_t is either V(s_t) (state baseline) or mean recent G (global baseline)

    Args:
        traj (list): Episode trajectory.
        alpha (float): Learning rate.
        gamma (float): Discount factor in G(tau).

    Returns:
        tuple: (theta_new, G)
    """
    if not traj:
        return get_theta(), 0.0

    theta = get_theta()
    G = _episode_return(traj, gamma=gamma)
    G_HISTORY.append(G)
    global_b = float(np.mean(G_HISTORY)) if len(G_HISTORY) > 0 else 0.0

    # accumulate sum_t grad log pi(a_t|s_t) per state position
    grad_sum = {}

    for step in traj:
        s = step["state"]
        a = int(step["action"])
        probs = np.asarray(step["probs"], dtype=np.float64)
        key = state_key(s)

        if key not in theta:
            theta[key] = np.zeros(4, dtype=np.float64)
        if key not in grad_sum:
            grad_sum[key] = np.zeros(4, dtype=np.float64)

        if use_state_baseline:
            b = BASELINE_V.get(key, 0.0)
        else:
            b = global_b

        advantage = G - b

        one_hot = np.zeros(4, dtype=np.float64)
        one_hot[a] = 1.0

        # grad log pi for softmax logits
        grad_log_pi = one_hot - probs

        grad_sum[key] += advantage * grad_log_pi

    # gradient ascent step
    for key, g in grad_sum.items():
        theta[key] += alpha * g

    if use_state_baseline:
        visited_keys = set(state_key(step["state"]) for step in traj)
        for key in visited_keys:
            old = BASELINE_V.get(key, 0.0)
            BASELINE_V[key] = (1.0 - beta_v) * old + beta_v * G
    
    set_theta(theta)
    return theta, G
