import os
import sys
import random
import json
import time

# --- ПАКЕТНЫЕ ИМПОРТЫ ---
# Ensure the parent directory is in sys.path if running as script
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Add Prj1 to sys.path
prj1_dir = os.path.join(current_dir, "Prj1")
if prj1_dir not in sys.path:
    sys.path.insert(0, prj1_dir)

try:
    import config
    from room_graph import build_random_room, DIRECTIONS
    from reinforce_cover import FloorCoverEnv
except ImportError as e:
    print(f"Import Error: {e}")
    sys.exit(1)

# Import Visualization
ENABLE_VISUALIZATION = config.ENABLE_VISUALIZATION
try:
    import pygame
    from visualization import init_pygame, draw_grid
except ImportError:
    print("Pygame not found or error importing visualization. Visualization disabled.")
    ENABLE_VISUALIZATION = False

# Import Plotting
HAVE_PLOT = False
try:
    import matplotlib
    matplotlib.use("Agg")  # No GUI backend
    from visualization_utils import plot_training_results
    HAVE_PLOT = True
except Exception as e:
    print(f"Plotting disabled: {e}")

# Paths
PRJ1_DIR = os.path.join(current_dir, "Prj1")
STATS_FILE = os.path.join(PRJ1_DIR, "training_stats_DFS.json")
PLOT_FILE  = os.path.join(PRJ1_DIR, "training_plot_DFS.png")


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

        # Identify valid moves
        valid_indices = [i for i, val in enumerate(action_mask) if val > 0.5]
        
        # Identify unvisited neighbors
        unvisited = []
        for idx in valid_indices:
            dr, dc = DIRECTIONS[idx]
            if (r + dr, c + dc) not in self.visited:
                unvisited.append(idx)

        # Logic: Prioritize unvisited -> Backtrack -> Random fallback
        if unvisited:
            action = random.choice(unvisited)
            self.stack.append((r, c)) 
            return action
        elif self.stack:
            back_pos = self.stack.pop()
            for idx in valid_indices:
                dr, dc = DIRECTIONS[idx]
                if (r + dr, c + dc) == back_pos:
                    return idx
        
        return random.choice(valid_indices) if valid_indices else 0


# --- JSON SAVE HELPER ---
def save_stats_json(stats_list, filename):
    try:
        with open(filename, 'w') as f:
            json.dump(stats_list, f, indent=4)
    except Exception as e:
        print(f"Error saving stats: {e}")


# --- MAIN LOOP ---
def main():
    print("--- STARTING HEURISTIC DFS AGENT ---")
    
    # Use config parameters
    NUM_EPISODES = 2000
    if hasattr(config, 'NUM_EPISODES'):
        NUM_EPISODES = config.NUM_EPISODES

    # Initialize Pygame
    screen = None
    font = None
    small_font = None
    tiny_font = None
    clock = None
    
    if ENABLE_VISUALIZATION:
        print("Visualization Enabled. Press SPACE to Pause/Resume, 'A' for Auto-Mode, 'R' to Reset Episode.")
        screen, font, small_font, tiny_font, clock = init_pygame()
    else:
        print("Visualization Disabled (Headless Mode).")

    training_history = []
    
    # Remove old stats file if it exists to start fresh
    if os.path.exists(STATS_FILE):
        os.remove(STATS_FILE)

    print(f"Statistics will be saved to: {STATS_FILE}")

    auto_mode = False # Start paused or manual step? Let's default to False (Paused) or True? 
    # Usually better to start paused or slow. Let's start paused.
    # Actually, headless runs fast. Visual runs interactive.
    
    for episode_idx in range(NUM_EPISODES):
        episode = episode_idx + 1
        
        # 1. Generate Room
        rng = random.Random(42 + episode)
        room = build_random_room(
            rows=config.GRID_SIZE, 
            cols=config.GRID_SIZE, 
            add_walls=True, 
            max_obstacle_ratio=config.OBSTACLE_PROB, 
            rng=rng
        )
        
        # Set Start Position
        start_pos = (1, 1)
        if hasattr(config, 'START_POSITION'):
            start_pos = config.START_POSITION
            
        # Validate start position
        sx, sy = start_pos
        if not (0 <= sx < room.rows and 0 <= sy < room.cols):
            start_pos = (1, 1)
            sx, sy = start_pos
        
        # Ensure start is floor
        room.set_floor(sx, sy)

        # 2. Initialize Environment and Agent
        env = FloorCoverEnv(room, max_steps=config.MAX_STEPS_PER_EPISODE)
        agent = GreedyDFSAgent()
        obs, info = env.reset(start=start_pos)
        agent.reset()

        # 3. Run Episode
        rewards = []
        done = False
        step = 0
        G = 0.0
        max_safety_steps = config.MAX_STEPS_PER_EPISODE * 2 
        
        reset_requested = False

        while not done and step < max_safety_steps:
            step_once = False
            manual_action = None

            # --- Event Handling ---
            if ENABLE_VISUALIZATION:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit()
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            pygame.quit()
                            sys.exit()
                        if event.key == pygame.K_SPACE:
                            if auto_mode:
                                auto_mode = False
                            else:
                                step_once = True
                        if event.key == pygame.K_a:
                            auto_mode = not auto_mode
                        if event.key == pygame.K_r:
                            reset_requested = True
                        
                        # Manual Controls (Optional, overrides agent)
                        # Up/Down/Left/Right 
                        # Note: Visual directions might map to internal indices
                        # 0: UP, 1: DOWN, 2: RIGHT, 3: LEFT (Check DIRECTIONS in room_graph)
                        # Usually: UP=(-1,0), DOWN=(1,0), RIGHT=(0,1), LEFT=(0,-1)
                        # Adjust mapping if needed. Assuming standard Pygame keys.
                        # For now, let's just control step.

                if reset_requested:
                    break

                # --- Control Logic ---
                should_step = auto_mode or step_once
                
                if not should_step:
                    # Draw and wait
                    vis_state = {
                        'agent_pos': info['agent_pos'],
                        'visited': info['visited_set'],
                    }
                    draw_grid(screen, font, small_font, tiny_font, env.room, vis_state, G, episode, True, step)
                    clock.tick(config.FPS)
                    continue
            
            # --- Agent Action ---
            mask = env.get_action_mask()
            action = agent.get_action(info, mask)
            
            # --- Environment Step ---
            obs, reward, done, _, info = env.step(action)
            rewards.append(reward)
            G += reward
            step += 1
            
            # --- Update Visualization ---
            if ENABLE_VISUALIZATION:
                vis_state = {
                    'agent_pos': info['agent_pos'],
                    'visited': info['visited_set'],
                }
                # If headless episodes passed (or always if we want to see it)
                # config.HEADLESS_EPISODES might be high, so maybe ignore it if we are running this script specifically for visualization?
                # User asked to "launch visualization", so we should show it.
                if episode > getattr(config, 'HEADLESS_EPISODES', -1): 
                     # Only skip if we are in the "fast" phase, but for this script, maybe we want to see it?
                     # Let's assume passed episodes are skipped.
                     pass
                
                # Check for events to keep UI responsive even in fast mode
                pygame.event.pump()
                
                draw_grid(screen, font, small_font, tiny_font, env.room, vis_state, G, episode, False, step)
                pygame.display.flip()
                
                if auto_mode and config.AUTO_STEP_DELAY > 0:
                    pygame.time.delay(config.AUTO_STEP_DELAY)

        # 4. Collect Stats
        total_reward = sum(rewards)
        n_visited = info["visited"]
        visited_ratio = n_visited / info["n_floor"] if info["n_floor"] > 0 else 0

        stats_entry = {
            'episode': episode,
            'reward': total_reward,
            'visited_ratio': visited_ratio
        }
        training_history.append(stats_entry)

        # 5. Log
        if episode % 100 == 0 or episode == 1:
            recent_rewards = [h['reward'] for h in training_history[-100:]]
            avg_reward = sum(recent_rewards) / len(recent_rewards)
            print(f"Episode {episode}: visited {n_visited}/{info['n_floor']}, avg_reward_100={avg_reward:.2f}")
            save_stats_json(training_history, STATS_FILE)

    # Final Save
    save_stats_json(training_history, STATS_FILE)
    print("\nDone! Full statistics saved.")
    
    if HAVE_PLOT:
        plot_training_results(training_history, PLOT_FILE)
        print(f"Plot saved to: {PLOT_FILE}")
        
    if ENABLE_VISUALIZATION:
        pygame.quit()

if __name__ == "__main__":
    main()
