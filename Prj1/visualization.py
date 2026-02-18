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
    y = TOP_MARGIN + (GRID_SIZE - 1 - j) * (CELL_SIZE + GAP)
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


    status = "⏸ WAITING (press SPACE)" if waiting else f"▶ Episode {episode}, Step {step}"
    txt = font.render(status, True, TEXT_COLOR)
    screen.blit(txt, (20, 15))


    reward_color = REWARD_COLOR_POSITIVE if G > 0 else REWARD_COLOR_NEGATIVE
    g_txt = font.render(f"G (Reward): {G:.1f}", True, reward_color)

    reward_bg = pygame.Surface((g_txt.get_width() + 20, g_txt.get_height() + 10))
    reward_bg.set_alpha(180)
    reward_bg.fill((40, 45, 60))
    screen.blit(reward_bg, (15, 48))
    screen.blit(g_txt, (20, 50))


    agent_pos = state.get('agent_pos', (0,0))
    visited_cells = state.get('visited', set())

    # Determine observed cells (neighbors)
    # If explicit neighbors are not provided, we can fallback to room.get_neighbors
    # but state['neighbors'] usually has the observed statuses.
    # For now, let's assume 'observed' cells are just the immediate neighbors + current cell
    # if not explicitly provided in a nice format.
    # In reinforce_cover.py, 'neighbors' key isn't standard in _get_info().
    # We'll adapt: if 'neighbors' in state, use it. Else calculate.





    observed_cells = set()

    observed_cells.add(agent_pos)
    for neighbor in room.get_neighbors(agent_pos[0], agent_pos[1]):
         observed_cells.add(neighbor)



    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            node = (i, j)
            x, y = grid_to_screen(i, j)


            is_agent = (node == agent_pos)
            is_observed = (node in observed_cells)


            is_floor = room.is_floor(i, j)
            is_obstacle = not is_floor
            is_visited = (node in visited_cells) and is_floor

            if is_agent:
                color = AGENT_COLOR
            elif is_observed:

                if is_obstacle:
                    color = OBSERVED_OBSTACLE_COLOR
                elif is_visited:
                    color = OBSERVED_VISITED_COLOR
                else:
                    color = OBSERVED_FREE_COLOR
            else:

                if is_obstacle:
                    color = OBSTACLE_COLOR
                elif is_visited:
                    color = VISITED_COLOR
                else:
                    color = FREE_COLOR


            pygame.draw.rect(screen, color, (x, y, CELL_SIZE, CELL_SIZE), border_radius=4)
            pygame.draw.rect(screen, GRID_LINE_COLOR, (x, y, CELL_SIZE, CELL_SIZE), 1, border_radius=4)


            if is_agent and action_probs is not None:
                cx, cy = x + CELL_SIZE // 2, y + CELL_SIZE // 2












                screen_dirs = [
                    (-1, 0),
                    (1, 0),
                    (0, -1),
                    (0, 1)
                ]

                for idx, (dx, dy) in enumerate(screen_dirs):
                    prob = action_probs[idx]
                    if prob > 0.01:

                        length = int((CELL_SIZE / 2 - 4) * (prob / 1.0))
                        end_x = cx + dx * length
                        end_y = cy + dy * length


                        prob_color = (255, 255, 0)
                        pygame.draw.line(screen, prob_color, (cx, cy), (end_x, end_y), width=3)


                        pygame.draw.circle(screen, prob_color, (end_x, end_y), 4)


    legend_x = MARGIN * 2 + GRID_SIZE * CELL_SIZE + (GRID_SIZE - 1) * GAP + 20
    legend_y_start = TOP_MARGIN + 20


    legend_title = small_font.render("LEGEND:", True, TEXT_COLOR)
    screen.blit(legend_title, (legend_x, legend_y_start))

    legend_items = [
        ("Agent", AGENT_COLOR),
        ("", None),
        ("Observed:", None),
        ("  + Free", OBSERVED_FREE_COLOR),
        ("  + Visited", OBSERVED_VISITED_COLOR),
        ("  + Obstacle", OBSERVED_OBSTACLE_COLOR),
        ("", None),
        ("Not observed:", None),
        ("  Visited", VISITED_COLOR),
        ("  Obstacle", OBSTACLE_COLOR),
        ("  Free", FREE_COLOR),
    ]

    current_y = legend_y_start + 35
    for label, color in legend_items:
        if color is None:

            if label:
                txt = tiny_font.render(label, True, (200, 210, 230))
                screen.blit(txt, (legend_x, current_y))
                current_y += 25
            else:

                current_y += 10
        else:

            pygame.draw.rect(screen, color, (legend_x, current_y, 22, 22), border_radius=3)
            pygame.draw.rect(screen, GRID_LINE_COLOR, (legend_x, current_y, 22, 22), 1, border_radius=3)


            txt = tiny_font.render(label, True, LEGEND_COLOR)
            screen.blit(txt, (legend_x + 30, current_y + 2))
            current_y += 30


    if action_probs is not None:
        current_y += 20
        probs_title = tiny_font.render("Current Policy:", True, (255, 255, 0))
        screen.blit(probs_title, (legend_x, current_y))
        current_y += 20



        dir_names = ["Left", "Right", "Up", "Down"]
        for i, p in enumerate(action_probs):
            p_txt = tiny_font.render(f"{dir_names[i]}: {p:.2f}", True, TEXT_COLOR)
            screen.blit(p_txt, (legend_x, current_y))
            current_y += 18


    controls_y = H - 60
    controls = tiny_font.render("[SPACE=step  A=auto  R=reset  ESC=quit]", True, LEGEND_COLOR)
    screen.blit(controls, (W // 2 - controls.get_width() // 2, controls_y))

    pygame.display.flip()