"""
Scene Graph Management
Contains functions for creating and managing the grid world graph.
"""

import networkx as nx
import numpy as np
from config import GRID_SIZE, OBSTACLE_PROB, START_POSITION


def create_scene_graph(obstacle_prob=OBSTACLE_PROB):
    """
    Create a grid graph with obstacles.
    The boundary is always obstacles, and start position is always free.
    
    Returns:
        nx.Graph: NetworkX graph representing the scene
    """
    graph = nx.Graph()
    
    def add_edge_no_dubl(g, cur_node, other_node):
        """Add edge only if it doesn't already exist"""
        if g.has_node(other_node):
            if not g.has_edge(cur_node, other_node):
                g.add_edge(cur_node, other_node)
    
    for i in range(-1, GRID_SIZE + 1):
        for j in range(-1, GRID_SIZE + 1):
            if i == -1 or i == GRID_SIZE or j == -1 or j == GRID_SIZE:
                is_obs = True
            elif (i, j) == START_POSITION:
                is_obs = False
            else:
                is_obs = np.random.random() < obstacle_prob
            
            graph.add_node((i, j), is_obstacle=is_obs, is_visited=False)
    
    for i in range(-1, GRID_SIZE + 1):
        for j in range(-1, GRID_SIZE + 1):
            cur_node = (i, j)
            neighbors = [(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)]
            for other_node in neighbors:
                add_edge_no_dubl(graph, cur_node, other_node)
    
    return graph


def reset_graph(graph):
    """
    Reset all 'is_visited' flags to False.
    
    Args:
        graph: NetworkX graph
    """
    nx.set_node_attributes(graph, False, 'is_visited')


def check_completion(graph):
    """
    Check if all non-obstacle cells have been visited.
    
    Args:
        graph: NetworkX graph
        
    Returns:
        bool: True if all free cells are visited, False otherwise
    """
    for node, data in graph.nodes(data=True):
        # Skip boundary nodes (outside main grid)
        x, y = node
        if x < 0 or x >= GRID_SIZE or y < 0 or y >= GRID_SIZE:
            continue
        
        # If it's not an obstacle and not visited, completion not achieved
        if not data.get('is_obstacle', False) and not data.get('is_visited', False):
            return False
    
    return True


def get_completion_stats(graph):
    """
    Get statistics about visited cells.
    
    Args:
        graph: NetworkX graph
        
    Returns:
        dict: Statistics about visited/total free cells
    """
    total_free = 0
    visited_free = 0
    
    for node, data in graph.nodes(data=True):
        x, y = node
        if x < 0 or x >= GRID_SIZE or y < 0 or y >= GRID_SIZE:
            continue
        
        if not data.get('is_obstacle', False):
            total_free += 1
            if data.get('is_visited', False):
                visited_free += 1
    
    return {
        'total_free': total_free,
        'visited': visited_free,
        'remaining': total_free - visited_free,
        'percentage': (visited_free / total_free * 100) if total_free > 0 else 0
    }
