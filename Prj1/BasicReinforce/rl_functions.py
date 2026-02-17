"""
RL Functions for Grid World Game
Contains state management, step sampling, and reward calculation logic.
"""

import numpy as np
from policy import policy
from config import REWARD_VISITED, REWARD_NEW, VERBOSE_CONSOLE


def get_state(graph, pos, steps=0):
    """
    Get state representation in fixed order, including agent pos and steps.
    
    Neighbors order:
    1) Current position
    2) Left (x-1, y)
    3) Up (x, y+1)
    4) Right (x+1, y)
    5) Down (x, y-1)
    
    Args:
        graph: NetworkX graph
        pos: Current position (x, y)
        steps: Number of steps taken (default: 0)
        
    Returns:
        dict: {
            'neighbors': list of tuples (x, y, is_obstacle, is_visited),
            'agent_pos': (x, y),
            'steps': int
        }
    """
    x, y = pos
    offsets = [
        (0, 0),   # Current
        (-1, 0),  # Left
        (0, 1),   # Up
        (1, 0),   # Right
        (0, -1)   # Down
    ]
    
    neighbors = []
    for dx, dy in offsets:
        node = (x + dx, y + dy)
        if graph.has_node(node):
            data = graph.nodes[node]
            neighbors.append((
                node[0], 
                node[1], 
                data.get('is_obstacle', True), 
                data.get('is_visited', False)
            ))
        else:
            # Outside graph - treat as obstacle
            neighbors.append((node[0], node[1], True, False))
    
    return {
        'neighbors': neighbors,
        'agent_pos': pos,
        'steps': steps
    }


def sample_step(graph, s, eps=0.0):
    """
    Sample action from policy and return new state.
    Marks current cell as visited.
    
    Args:
        graph: NetworkX graph
        s: Current state dict
        eps (float): Exploration probability in [0, 1]
    Returns:
        dict: New state after taking action
    """
    neighbors = s['neighbors']
    current_node = neighbors[0]  # (x, y, obs, visited)
    old_pos = (current_node[0], current_node[1])
    
    # Mark current cell as visited
    graph.nodes[old_pos]['is_visited'] = True
    
    pol = policy(s)


    # valid actions among [Left, Up, Right, Down]
    valid_mask = np.array([not n[2] for n in neighbors[1:]], dtype=bool)
    valid_idx = np.where(valid_mask)[0]

    # epsilon exploration
    if np.random.rand() < eps:
        if len(valid_idx) > 0:
            idx = np.random.choice(valid_idx)
            pol_used = np.zeros_like(pol, dtype=np.float64)
            pol_used[valid_idx] = 1.0 / len(valid_idx)   # uniform over valid
        else:
            idx = np.random.choice(len(neighbors) - 1)
            pol_used = np.ones_like(pol, dtype=np.float64) / len(pol)
    else:
        idx = np.random.choice(len(neighbors) - 1, p=pol)
        pol_used = pol
        
    chosen = neighbors[idx + 1]
    
    new_pos = (chosen[0], chosen[1])
    
    # If obstacle, stay in place
    if graph.nodes[new_pos].get('is_obstacle', False):
        new_pos = old_pos
        
    if VERBOSE_CONSOLE:
        directions = ["Left", "Up", "Right", "Down"]
        print(f"  [sample_step] Chosen action: {directions[idx]} -> New Pos: {new_pos}")
    
    # Update steps count
    new_steps = s['steps'] + 1
    new_state = get_state(graph, new_pos, new_steps)
    return new_state, idx, pol_used



def reward(graph, s):
    """
    Calculate reward for current state.
    
    Args:
        graph: NetworkX graph
        s: Current state dict
        
    Returns:
        float: Reward value
    """
    curr_pos = s['agent_pos']
    
    if graph.nodes[curr_pos].get('is_visited', False):
        return REWARD_VISITED
    else:
        return REWARD_NEW
