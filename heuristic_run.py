# import os
# import sys
# import random
# import json

# # --- ПАКЕТНЫЕ ИМПОРТЫ (важно для `python -m Prj1.heuristic_run`) ---
# from Prj1 import config
# from Prj1.room_graph import build_random_room, DIRECTIONS
# from Prj1.reinforce_cover import FloorCoverEnv

# # принудительно без окна (headless baseline)
# config.ENABLE_VISUALIZATION = False

# # plotting (как в RL)
# try:
#     import matplotlib
#     matplotlib.use("Agg")  # чтобы без окна
#     from Prj1.visualization_utils import plot_training_results
#     HAVE_PLOT = True
# except Exception as e:
#     print(f"Plotting disabled: {e}")
#     HAVE_PLOT = False

# # пути к файлам ВНУТРИ папки Prj1
# PRJ1_DIR = os.path.dirname(os.path.abspath(__file__))
# STATS_FILE = os.path.join(PRJ1_DIR, "training_stats_DFS.json")
# PLOT_FILE  = os.path.join(PRJ1_DIR, "training_plot_DFS.png")

# # --- 1. PATH SETUP ---
# current_dir = os.path.dirname(os.path.abspath(__file__))
# prj1_dir = os.path.join(current_dir, "Prj1")
# if prj1_dir not in sys.path:
#     sys.path.append(prj1_dir)
# try:
#     import matplotlib
#     matplotlib.use("Agg")  
#     from visualization_utils import plot_training_results
#     HAVE_PLOT = True
# except Exception as e:
#     print(f"Plotting disabled: {e}")
#     HAVE_PLOT = False
# # --- 2. IMPORT PROJECT MODULES ---
# try:
#     import Prj1.config as config
#     # Force disable visualization for speed
#     config.ENABLE_VISUALIZATION = False 
    
#     from Prj1.room_graph import build_random_room, DIRECTIONS
#     from Prj1.reinforce_cover import FloorCoverEnv
# except ImportError as e:
#     print(f"Import Error: {e}")
#     sys.exit(1)

# # File to save statistics (Same format as RL)
# STATS_FILE = os.path.join(prj1_dir, "training_stats_DFS.json")
# PLOT_FILE = os.path.join(prj1_dir, "training_plot_DFS.png")

# # --- 3. DFS AGENT ---
# class GreedyDFSAgent:
#     def __init__(self):
#         self.stack = []
#         self.visited = set()

#     def reset(self):
#         self.stack = []
#         self.visited = set()

#     def get_action(self, info, action_mask):
#         r, c = info['agent_pos']
        
#         # Sync visited set
#         if 'visited_set' in info:
#             self.visited = info['visited_set']
#         else:
#             self.visited.add((r, c))

#         # Identify valid moves
#         valid_indices = [i for i, val in enumerate(action_mask) if val > 0.5]
        
#         # Identify unvisited neighbors
#         unvisited = []
#         for idx in valid_indices:
#             dr, dc = DIRECTIONS[idx]
#             if (r + dr, c + dc) not in self.visited:
#                 unvisited.append(idx)

#         # Logic: Prioritize unvisited -> Backtrack -> Random fallback
#         if unvisited:
#             action = random.choice(unvisited)
#             self.stack.append((r, c)) 
#             return action
#         elif self.stack:
#             back_pos = self.stack.pop()
#             for idx in valid_indices:
#                 dr, dc = DIRECTIONS[idx]
#                 if (r + dr, c + dc) == back_pos:
#                     return idx
        
#         return random.choice(valid_indices) if valid_indices else 0

# # --- 4. JSON SAVE HELPER ---
# def save_stats_json(stats_list, filename):
#     with open(filename, 'w') as f:
#         json.dump(stats_list, f, indent=4)

# # --- 5. MAIN LOOP (RL-LIKE BASELINE) ---
# def main():
#     print("--- STARTING DFS BASELINE (RL-LIKE SETUP) ---")
    
#     # Parameters (Must match RL config)
#     NUM_EPISODES = 2000
#     training_history = []
    
#     # Remove old stats file if it exists to start fresh
#     if os.path.exists(STATS_FILE):
#         os.remove(STATS_FILE)

#     print(f"Statistics will be saved to: {STATS_FILE}")

#     for episode_idx in range(NUM_EPISODES):
#         episode = episode_idx + 1
        
#         # 1. Generate Room (Same seed logic as RL)
#         rng = random.Random(42 + episode)
#         room = build_random_room(
#             rows=config.GRID_SIZE, 
#             cols=config.GRID_SIZE, 
#             add_walls=True, 
#             max_obstacle_ratio=config.OBSTACLE_PROB, 
#             rng=rng
#         )
        
#         # Set Start Position
#         sx, sy = config.START_POSITION
#         if not (0 <= sx < room.rows and 0 <= sy < room.cols):
#             sx, sy = 1, 1
#         room.set_floor(sx, sy)

#         # 2. Initialize Environment and Agent
#         env = FloorCoverEnv(room, max_steps=config.MAX_STEPS_PER_EPISODE)
#         agent = GreedyDFSAgent()
#         obs, info = env.reset(start=(sx, sy))
#         agent.reset()

#         # 3. Run Episode
#         rewards = []
#         done = False
#         step = 0
#         max_safety_steps = config.MAX_STEPS_PER_EPISODE * 2 # infinite loop protection

#         while not done and step < max_safety_steps:
#             mask = env.get_action_mask()
#             action = agent.get_action(info, mask)
#             obs, reward, done, _, info = env.step(action)
#             rewards.append(reward)
#             step += 1

#         # 4. Collect Stats
#         total_reward = sum(rewards)
#         n_visited = info["visited"]
#         visited_ratio = n_visited / info["n_floor"] if info["n_floor"] > 0 else 0

#         # Data structure identical to RL stats
#         stats_entry = {
#             'episode': episode,
#             'reward': total_reward,
#             'visited_ratio': visited_ratio
#         }
#         training_history.append(stats_entry)

#         # 5. Log to console (Every 100 episodes)
#         if episode % 100 == 0:
#             # Calculate average over last 100 episodes
#             recent_rewards = [h['reward'] for h in training_history[-100:]]
#             avg_reward = sum(recent_rewards) / len(recent_rewards)
            
#             print(f"Episode {episode}: visited {n_visited}/{info['n_floor']}, avg_reward_100={avg_reward:.2f}")
            
#             # Intermediate save
#             save_stats_json(training_history, STATS_FILE)

#     # Final Save
#     save_stats_json(training_history, STATS_FILE)
#     print("\nDone! Full statistics saved to JSON.")
#     if HAVE_PLOT:
#         plot_training_results(training_history, PLOT_FILE)
#         print(f"Plot saved to: {PLOT_FILE}")

# if __name__ == "__main__":
#     main()

import sys
import os
import random
import json
import csv
import importlib

import numpy as np


# ---------- utils ----------
def safe_import(module_names, attr=None):
    """
    Try import module from list. Optionally get attr/class from module.
    """
    last_err = None
    for name in module_names:
        try:
            mod = importlib.import_module(name)
            return getattr(mod, attr) if attr else mod
        except Exception as e:
            last_err = e
            continue
    print(f"CRITICAL ERROR: Could not import {module_names}" + (f".{attr}" if attr else ""))
    if last_err:
        print("Last error:", last_err)
    return None


# ---------- PATH ----------
current_dir = os.path.dirname(os.path.abspath(__file__))
prj1_dir = os.path.join(current_dir, "Prj1")
if prj1_dir not in sys.path:
    sys.path.append(prj1_dir)


# ---------- CONFIG ----------
config = safe_import(["Prj1.config", "config"])
if not config:
    print("Config not found, abort.")
    sys.exit(1)

GRID_SIZE = getattr(config, "GRID_SIZE", 10)
OBSTACLE_PROB = getattr(config, "OBSTACLE_PROB", 0.1)
START_POSITION = getattr(config, "START_POSITION", (1, 1))
MAX_STEPS_PER_EPISODE = getattr(config, "MAX_STEPS_PER_EPISODE", 200)

# Сколько эпизодов гонять (как у тебя HEADLESS_EPISODES в config)
DEFAULT_EPISODES = getattr(config, "HEADLESS_EPISODES", 2000)

# Управление через env:
#   HEADLESS=1  -> без pygame/окна
#   EPISODES=2000 -> число эпизодов
#   RUN_TAG=HEURISTIC_DFS -> суффикс файлов
HEADLESS = os.environ.get("HEADLESS", "1") == "1"
NUM_EPISODES = int(os.environ.get("EPISODES", str(DEFAULT_EPISODES)))
RUN_TAG = os.environ.get("RUN_TAG", "HEURISTIC_DFS")

PLOT_TRAINING_CURVE = getattr(config, "PLOT_TRAINING_CURVE", True)
ENABLE_LOGGING = getattr(config, "ENABLE_LOGGING", True)


# ---------- PROJECT IMPORTS ----------
room_graph_mod = safe_import(["Prj1.room_graph", "room_graph"])
if not room_graph_mod:
    sys.exit(1)

build_random_room = room_graph_mod.build_random_room
DIRECTIONS = room_graph_mod.DIRECTIONS

FloorCoverEnv = safe_import(
    ["Prj1.reinforce_cover", "reinforce_cover", "Prj1.heuristic_main"],
    "FloorCoverEnv",
)
if not FloorCoverEnv:
    print("FloorCoverEnv not found!")
    sys.exit(1)

# visualization_utils (как в RL) — для json/plot
plot_training_results = None
save_stats_json = None
try:
    vu = safe_import(["Prj1.visualization_utils", "visualization_utils"])
    if vu:
        plot_training_results = getattr(vu, "plot_training_results", None)
        save_stats_json = getattr(vu, "save_stats_json", None)
except Exception:
    pass


# ---------- DFS AGENT ----------
class GreedyDFSAgent:
    def __init__(self):
        self.stack = []
        self.visited = set()

    def reset(self):
        self.stack = []
        self.visited = set()

    def get_action(self, info, action_mask):
        r, c = info["agent_pos"]

        # берём visited из env (если есть)
        if "visited_set" in info:
            self.visited = info["visited_set"]
        else:
            self.visited.add((r, c))

        valid = [i for i, v in enumerate(action_mask) if v > 0.5]

        # ищем непосещённых соседей
        unvisited = []
        for a in valid:
            dr, dc = DIRECTIONS[a]
            nr, nc = r + dr, c + dc
            if (nr, nc) not in self.visited:
                unvisited.append(a)

        if unvisited:
            a = random.choice(unvisited)
            self.stack.append((r, c))
            return a

        if self.stack:
            back = self.stack.pop()
            for a in valid:
                dr, dc = DIRECTIONS[a]
                if (r + dr, c + dc) == back:
                    return a

        return random.choice(valid) if valid else 0


def run_one_episode(env, agent, start_pos):
    obs, info = env.reset(start=start_pos)
    agent.reset()

    total_reward = 0.0
    steps = 0
    done = False

    # ВАЖНО: env может не увеличивать step_count при ударе в стену,
    # поэтому ограничиваем внешним steps < env.max_steps
    while (not done) and (steps < env.max_steps):
        mask = env.get_action_mask()
        action = agent.get_action(info, mask)
        obs, reward, done, _, info = env.step(action)
        total_reward += float(reward)
        steps += 1

    return steps, total_reward, info


def fallback_save_json(history, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(history, f, indent=2)


def main():
    print(f"--- HEURISTIC BASELINE: {RUN_TAG} ---")
    print(f"HEADLESS={HEADLESS}, EPISODES={NUM_EPISODES}")

    # Выходные файлы (как у RL, но с тэгом)
    stats_json = os.path.join(prj1_dir, f"training_stats_{RUN_TAG}.json")
    plot_png = os.path.join(prj1_dir, f"training_plot_{RUN_TAG}.png")
    results_csv = os.path.join(prj1_dir, f"heuristic_results_{RUN_TAG}.csv")

    os.makedirs(prj1_dir, exist_ok=True)

    # training_history — один-в-один как в RL
    training_history = []

    # CSV лог
    csv_f = open(results_csv, "w", newline="", buffering=1)
    writer = csv.writer(csv_f)
    writer.writerow(["episode", "steps", "reward", "visited", "n_floor", "visited_ratio"])

    agent = GreedyDFSAgent()

    sx, sy = START_POSITION

    for episode in range(1, NUM_EPISODES + 1):
        rng = random.Random(42 + episode)
        room = build_random_room(
            rows=GRID_SIZE,
            cols=GRID_SIZE,
            add_walls=True,
            max_obstacle_ratio=OBSTACLE_PROB,
            rng=rng,
        )

        # гарантируем, что старт — floor
        if 0 <= sx < room.rows and 0 <= sy < room.cols:
            room.set_floor(sx, sy)
            start = (sx, sy)
        else:
            start = (1, 1)
            if 0 <= start[0] < room.rows and 0 <= start[1] < room.cols:
                room.set_floor(start[0], start[1])

        env = FloorCoverEnv(room, max_steps=MAX_STEPS_PER_EPISODE)

        steps, total_reward, info = run_one_episode(env, agent, start)

        visited = int(info.get("visited", 0))
        n_floor = int(info.get("n_floor", 0))
        visited_ratio = (visited / n_floor) if n_floor > 0 else 0.0

        # JSON entry — как у RL
        training_history.append({
            "episode": episode,
            "reward": float(total_reward),
            "visited_ratio": float(visited_ratio),
        })

        # CSV row
        writer.writerow([episode, steps, float(total_reward), visited, n_floor, float(visited_ratio)])

        # консольный лог как в RL (avg over last 100)
        if episode % 100 == 0 or episode == 1:
            recent = training_history[-100:]
            avg_reward = sum(x["reward"] for x in recent) / len(recent)
            print(f"Episode {episode}: visited {visited}/{n_floor}, avg_reward_100={avg_reward:.2f}")

        # периодически сохраняем json, чтобы не потерять прогресс
        if ENABLE_LOGGING and (episode % 100 == 0):
            if save_stats_json is not None:
                save_stats_json(training_history, stats_json)
            else:
                fallback_save_json(training_history, stats_json)

    csv_f.close()

    # финальное сохранение JSON
    if ENABLE_LOGGING:
        if save_stats_json is not None:
            save_stats_json(training_history, stats_json)
        else:
            fallback_save_json(training_history, stats_json)

    # PNG график (если есть visualization_utils как в RL)
    if PLOT_TRAINING_CURVE and (plot_training_results is not None):
        try:
            plot_training_results(training_history, plot_png)
        except Exception as e:
            print("Plot failed:", e)

    print("\nDONE.")
    print("CSV :", results_csv)
    print("JSON:", stats_json)
    if PLOT_TRAINING_CURVE and (plot_training_results is not None):
        print("PNG :", plot_png)
    else:
        print("PNG : (not generated — visualization_utils/plot missing)")


if __name__ == "__main__":
    main()
