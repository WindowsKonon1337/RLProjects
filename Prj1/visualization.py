"""
Pygame Visualization for Grid World RL Game
All rendering and drawing logic.
"""

import pygame
from config import *


def init_pygame():
    """
    Initialize pygame and create window.
    
    Returns:
        tuple: (screen, font, small_font, tiny_font, clock)
    """
    pygame.init()
    
    W = MARGIN * 2 + GRID_SIZE * CELL_SIZE + (GRID_SIZE - 1) * GAP + LEGEND_WIDTH
    H = TOP_MARGIN + MARGIN + GRID_SIZE * CELL_SIZE + (GRID_SIZE - 1) * GAP + INFO_HEIGHT
    
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("Grid World RL")
    
    font = pygame.font.SysFont("monospace", 20, bold=True)
    small_font = pygame.font.SysFont("monospace", 15)
    tiny_font = pygame.font.SysFont("monospace", 13)
    clock = pygame.time.Clock()
    
    return screen, font, small_font, tiny_font, clock


def grid_to_screen(i, j):
    """
    Convert grid coordinates to screen coordinates.
    
    Args:
        i, j: Grid coordinates
        
    Returns:
        tuple: (x, y) screen coordinates
    """
    x = MARGIN + i * (CELL_SIZE + GAP)
    y = TOP_MARGIN + (GRID_SIZE - 1 - j) * (CELL_SIZE + GAP)  # Flip Y, use TOP_MARGIN
    return x, y


def draw_grid(screen, font, small_font, tiny_font, room, state, G, episode, waiting, step, action_probs=None):
    """
    Draw the entire game state.
    
    Args:
        screen: Pygame screen
        font, small_font, tiny_font: Font objects
        room: RoomGraph object from the environment
        state: Current state dictionary or object containing agent position and visited info
               Expected keys: 'agent_pos', 'visited', optionally 'neighbors' for observed cells
        G: Cumulative reward
        episode: Episode number
        waiting: Whether waiting for next episode
        step: Step number
        action_probs: Optional list of 4 probabilities [Left, Up, Right, Down]
    """
    screen.fill(BG_COLOR)
    
    W = screen.get_width()
    H = screen.get_height()
    
    # Draw info bar at top
    status = "⏸ WAITING (press SPACE)" if waiting else f"▶ Episode {episode}, Step {step}"
    txt = font.render(status, True, TEXT_COLOR)
    screen.blit(txt, (20, 15))
    
    # CUMULATIVE REWARD - VERY VISIBLE WITH BACKGROUND
    reward_color = REWARD_COLOR_POSITIVE if G > 0 else REWARD_COLOR_NEGATIVE
    g_txt = font.render(f"G (Reward): {G:.1f}", True, reward_color)
    # Draw semi-transparent background for reward
    reward_bg = pygame.Surface((g_txt.get_width() + 20, g_txt.get_height() + 10))
    reward_bg.set_alpha(180)
    reward_bg.fill((40, 45, 60))
    screen.blit(reward_bg, (15, 48))
    screen.blit(g_txt, (20, 50))
    
    # Get agent position and observed cells
    agent_pos = state.get('agent_pos', (0,0))
    visited_cells = state.get('visited', set())
    
    # Determine observed cells (neighbors)
    # If explicit neighbors are not provided, we can fallback to room.get_neighbors
    # but state['neighbors'] usually has the observed statuses.
    # For now, let's assume 'observed' cells are just the immediate neighbors + current cell
    # if not explicitly provided in a nice format. 
    # In reinforce_cover.py, 'neighbors' key isn't standard in _get_info().
    # We'll adapt: if 'neighbors' in state, use it. Else calculate.
    
    # Based on reinforce_cover.py: _get_info() returns {"visited": count, "n_floor": count}
    # We need to pass the actual set of visited cells and agent position to this function.
    # We'll rely on the caller to augment state dict with 'visited' set and 'agent_pos'.
    
    observed_cells = set()
    # Add current position and neighbors to observed for visualization effect
    observed_cells.add(agent_pos)
    for neighbor in room.get_neighbors(agent_pos[0], agent_pos[1]):
         observed_cells.add(neighbor)

    
    # Draw grid cells (MAIN FIELD)
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            node = (i, j)
            x, y = grid_to_screen(i, j)
            
            # Determine cell color based on priority/state
            is_agent = (node == agent_pos)
            is_observed = (node in observed_cells)
            
            # Check properties using RoomGraph methods
            is_floor = room.is_floor(i, j)
            is_obstacle = not is_floor
            is_visited = (node in visited_cells) and is_floor
            
            if is_agent:
                color = AGENT_COLOR
            elif is_observed:
                # Observed cells get special colors
                if is_obstacle:
                    color = OBSERVED_OBSTACLE_COLOR  # Dark red
                elif is_visited:
                    color = OBSERVED_VISITED_COLOR  # Dark orange
                else:
                    color = OBSERVED_FREE_COLOR  # Dark yellow
            else:
                # Non-observed cells
                if is_obstacle:
                    color = OBSTACLE_COLOR  # Purple
                elif is_visited:
                    color = VISITED_COLOR  # Green
                else:
                    color = FREE_COLOR  # Gray
            
            # Draw cell
            pygame.draw.rect(screen, color, (x, y, CELL_SIZE, CELL_SIZE), border_radius=4)
            pygame.draw.rect(screen, GRID_LINE_COLOR, (x, y, CELL_SIZE, CELL_SIZE), 1, border_radius=4)
            
            # Draw action probabilities on agent cell
            if is_agent and action_probs is not None:
                cx, cy = x + CELL_SIZE // 2, y + CELL_SIZE // 2
                # directions: Left (-1,0), Up (0,1), Right (1,0), Down (0,-1)
                # coords: x increases right, y increases down (screen)
                # Grid Up is +j -> Screen Up (y decrease)
                # Grid Down is -j -> Screen Down (y increase)
                
                # VISUALIZATION MAPPING FIX:
                # Based on grid_to_screen and room_graph.DIRECTIONS:
                # Action 0 (UP, dr=-1) -> i decreases -> Screen LEFT
                # Action 1 (DOWN, dr=+1) -> i increases -> Screen RIGHT
                # Action 2 (RIGHT, dc=+1) -> j increases -> Screen UP
                # Action 3 (LEFT, dc=-1) -> j decreases -> Screen DOWN
                
                screen_dirs = [
                    (-1, 0), # 0: Code UP -> Visual LEFT
                    (1, 0),  # 1: Code DOWN -> Visual RIGHT
                    (0, -1), # 2: Code RIGHT -> Visual UP
                    (0, 1)   # 3: Code LEFT -> Visual DOWN
                ]
                
                for idx, (dx, dy) in enumerate(screen_dirs):
                    prob = action_probs[idx]
                    if prob > 0.01:
                        # Draw line proportional to prob
                        length = int((CELL_SIZE / 2 - 4) * (prob / 1.0)) # Scale max length
                        end_x = cx + dx * length
                        end_y = cy + dy * length
                        
                        # Use a bright color for probability indicators
                        prob_color = (255, 255, 0)
                        pygame.draw.line(screen, prob_color, (cx, cy), (end_x, end_y), width=3)
                        
                        # Draw small circle at end
                        pygame.draw.circle(screen, prob_color, (end_x, end_y), 4)

    # LEGEND - RIGHT SIDE
    legend_x = MARGIN * 2 + GRID_SIZE * CELL_SIZE + (GRID_SIZE - 1) * GAP + 20
    legend_y_start = TOP_MARGIN + 20
    
    # Title
    legend_title = small_font.render("LEGEND:", True, TEXT_COLOR)
    screen.blit(legend_title, (legend_x, legend_y_start))
    
    legend_items = [
        ("Agent", AGENT_COLOR),
        ("", None),  # Separator
        ("Observed:", None),  # Category
        ("  + Free", OBSERVED_FREE_COLOR),
        ("  + Visited", OBSERVED_VISITED_COLOR),
        ("  + Obstacle", OBSERVED_OBSTACLE_COLOR),
        ("", None),  # Separator
        ("Not observed:", None),  # Category
        ("  Visited", VISITED_COLOR),
        ("  Obstacle", OBSTACLE_COLOR),
        ("  Free", FREE_COLOR),
    ]
    
    current_y = legend_y_start + 35
    for label, color in legend_items:
        if color is None:
            # Category or separator
            if label:
                txt = tiny_font.render(label, True, (200, 210, 230))
                screen.blit(txt, (legend_x, current_y))
                current_y += 25
            else:
                # Just spacing
                current_y += 10
        else:
            # Draw color box
            pygame.draw.rect(screen, color, (legend_x, current_y, 22, 22), border_radius=3)
            pygame.draw.rect(screen, GRID_LINE_COLOR, (legend_x, current_y, 22, 22), 1, border_radius=3)
            
            # Draw label
            txt = tiny_font.render(label, True, LEGEND_COLOR)
            screen.blit(txt, (legend_x + 30, current_y + 2))
            current_y += 30
            
    # Draw current probabilities text in legend as well
    if action_probs is not None:
        current_y += 20
        probs_title = tiny_font.render("Current Policy:", True, (255, 255, 0))
        screen.blit(probs_title, (legend_x, current_y))
        current_y += 20
        
        # Labels match Action Indices: 0, 1, 2, 3
        # 0=UP(Left), 1=DOWN(Right), 2=RIGHT(Up), 3=LEFT(Down)
        dir_names = ["Left", "Right", "Up", "Down"] 
        for i, p in enumerate(action_probs):
            p_txt = tiny_font.render(f"{dir_names[i]}: {p:.2f}", True, TEXT_COLOR)
            screen.blit(p_txt, (legend_x, current_y))
            current_y += 18
    
    # Controls info at bottom
    controls_y = H - 60
    controls = tiny_font.render("[SPACE=step  A=auto  R=reset  ESC=quit]", True, LEGEND_COLOR)
    screen.blit(controls, (W // 2 - controls.get_width() // 2, controls_y))
    
    pygame.display.flip()
