import sys
import os
import random
import time
import numpy as np
import pygame
import importlib

# --- БЕЗОПАСНЫЙ ИМПОРТ ---
def safe_import(module_names, class_name=None):
    for mod_name in module_names:
        try:
            module = importlib.import_module(mod_name)
            if class_name:
                return getattr(module, class_name)
            return module
        except (ImportError, AttributeError):
            continue
    print(f"CRITICAL ERROR: Could not find module/class: {module_names}")
    return None
current_dir = os.path.dirname(os.path.abspath(__file__))
prj1_dir = os.path.join(current_dir, "Prj1")
if prj1_dir not in sys.path:
    sys.path.append(prj1_dir)
# 1. Config
config = safe_import(["Prj1.config", "config"])
if config:
    GRID_SIZE = config.GRID_SIZE
    OBSTACLE_PROB = config.OBSTACLE_PROB
    START_POSITION = config.START_POSITION
    MAX_STEPS_PER_EPISODE = config.MAX_STEPS_PER_EPISODE
    FPS = getattr(config, 'FPS', 60)
    AUTO_STEP_DELAY = getattr(config, 'AUTO_STEP_DELAY', 50)
    ENABLE_VISUALIZATION = getattr(config, 'ENABLE_VISUALIZATION', True)
else:
    print("Config not found, using defaults.")
    GRID_SIZE = 10
    START_POSITION = (1, 1)
    MAX_STEPS_PER_EPISODE = 200
    FPS = 60
    AUTO_STEP_DELAY = 50
    ENABLE_VISUALIZATION = True

# 2. Room Graph
room_graph_mod = safe_import(["Prj1.room_graph", "room_graph"])
if room_graph_mod:
    build_random_room = room_graph_mod.build_random_room
    DIRECTIONS = room_graph_mod.DIRECTIONS
else:
    sys.exit(1)

# 3. Visualization
vis_mod = safe_import(["Prj1.visualization", "visualization"])
if vis_mod:
    init_pygame = vis_mod.init_pygame
    draw_grid = vis_mod.draw_grid
else:
    ENABLE_VISUALIZATION = False

# 4. Environment (FloorCoverEnv)
FloorCoverEnv = safe_import(
    ["Prj1.reinforce_cover", "reinforce_cover", "Prj1.heuristic_main"], 
    "FloorCoverEnv"
)
if not FloorCoverEnv:
    print("FloorCoverEnv not found!")
    sys.exit(1)


# --- DFS AGENT ---
class GreedyDFSAgent:
    def __init__(self):
        self.stack = []
        self.visited = set()

    def reset(self):
        self.stack = []
        self.visited = set()

    def get_action(self, info, action_mask):
        r, c = info['agent_pos']
        
        # Sync visited set
        if 'visited_set' in info:
            self.visited = info['visited_set']
        else:
            self.visited.add((r, c))

        # 1. Valid moves
        valid_indices = [i for i, val in enumerate(action_mask) if val > 0.5]
        
        # 2. Unvisited neighbors
        unvisited = []
        for idx in valid_indices:
            dr, dc = DIRECTIONS[idx]
            neighbor = (r + dr, c + dc)
            if neighbor not in self.visited:
                unvisited.append(idx)

        # Logic
        if unvisited:
            # Go forward (Greedy)
            action = random.choice(unvisited)
            self.stack.append((r, c)) 
            return action
        elif self.stack:
            # Backtrack
            back_pos = self.stack.pop()
            for idx in valid_indices:
                dr, dc = DIRECTIONS[idx]
                if (r + dr, c + dc) == back_pos:
                    return idx
        
        # Stuck -> Random
        return random.choice(valid_indices) if valid_indices else 0


# --- MAIN ---
def main():
    print("Starting Heuristic Baseline...")
    
    # Log file
    log_file = open("heuristic_results.txt", "w")
    log_file.write("Episode,Steps,Reward,Visited,TotalFloor\n")

    if ENABLE_VISUALIZATION:
        screen, font, small_font, tiny_font, clock = init_pygame()
    
    # Initial Room
    rng = random.Random(42)
    room = build_random_room(rows=GRID_SIZE, cols=GRID_SIZE, max_obstacle_ratio=OBSTACLE_PROB, rng=rng)
    
    # Ensure start is valid
    sx, sy = START_POSITION
    if 0 <= sx < room.rows and 0 <= sy < room.cols:
        room.set_floor(sx, sy)

    env = FloorCoverEnv(room, max_steps=MAX_STEPS_PER_EPISODE)
    agent = GreedyDFSAgent()
    
    obs, info = env.reset(start=START_POSITION)
    agent.reset()
    
    running = True
    auto = False
    step_once = False
    episode = 1
    total_reward = 0
    step = 0
    waiting = False

    while running:
        step_once = False
        
        # Events
        if ENABLE_VISUALIZATION:
            for event in pygame.event.get():
                if event.type == pygame.QUIT: running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE: running = False
                    if event.key == pygame.K_a: auto = not auto
                    if event.key == pygame.K_r: 
                        obs, info = env.reset(start=START_POSITION)
                        agent.reset()
                        step = 0; total_reward = 0; waiting = False
                    if event.key == pygame.K_SPACE:
                        if waiting:
                            episode += 1
                            rng = random.Random(42 + episode)
                            room = build_random_room(rows=GRID_SIZE, cols=GRID_SIZE, max_obstacle_ratio=OBSTACLE_PROB, rng=rng)
                            room.set_floor(sx, sy)
                            env = FloorCoverEnv(room, max_steps=MAX_STEPS_PER_EPISODE)
                            obs, info = env.reset(start=START_POSITION)
                            agent.reset()
                            step = 0; total_reward = 0; waiting = False
                        else:
                            step_once = True
        else:
            auto = True # Run fast if headless

        # Step
        if not waiting and (auto or step_once):
            mask = env.get_action_mask()
            action = agent.get_action(info, mask)
            
            obs, reward, done, _, info = env.step(action)
            total_reward += reward
            step += 1
            
            if done:
                waiting = True
                auto = False
                print(f"Ep {episode}: Reward {total_reward:.2f}, Steps {step}, Visited {info['visited']}/{info['n_floor']}")
                log_file.write(f"{episode},{step},{total_reward},{info['visited']},{info['n_floor']}\n")
                log_file.flush()

        # Draw
        if ENABLE_VISUALIZATION:
            try:
                vis_state = {'agent_pos': info['agent_pos'], 'visited': info.get('visited_set', set())}
                draw_grid(screen, font, small_font, tiny_font, env.room, vis_state, total_reward, episode, waiting, step, action_probs=None)
            except TypeError:
                vis_state = {'agent_pos': info['agent_pos'], 'visited': info.get('visited_set', set())}
                draw_grid(screen, font, small_font, tiny_font, env.room, vis_state, total_reward, episode, waiting, step)

            pygame.display.flip()
            clock.tick(FPS)
            if auto: pygame.time.delay(AUTO_STEP_DELAY)

    log_file.close()
    if ENABLE_VISUALIZATION: pygame.quit()

if __name__ == "__main__":
    main()