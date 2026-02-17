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

DEFAULT_EPISODES = getattr(config, "HEADLESS_EPISODES", 2000)

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

        if "visited_set" in info:
            self.visited = info["visited_set"]
        else:
            self.visited.add((r, c))

        valid = [i for i, v in enumerate(action_mask) if v > 0.5]

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

    stats_json = os.path.join(prj1_dir, f"training_stats_{RUN_TAG}.json")
    plot_png = os.path.join(prj1_dir, f"training_plot_{RUN_TAG}.png")
    results_csv = os.path.join(prj1_dir, f"heuristic_results_{RUN_TAG}.csv")

    os.makedirs(prj1_dir, exist_ok=True)

    training_history = []

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

        training_history.append({
            "episode": episode,
            "reward": float(total_reward),
            "visited_ratio": float(visited_ratio),
        })

        writer.writerow([episode, steps, float(total_reward), visited, n_floor, float(visited_ratio)])

        if episode % 100 == 0 or episode == 1:
            recent = training_history[-100:]
            avg_reward = sum(x["reward"] for x in recent) / len(recent)
            print(f"Episode {episode}: visited {visited}/{n_floor}, avg_reward_100={avg_reward:.2f}")

        if ENABLE_LOGGING and (episode % 100 == 0):
            if save_stats_json is not None:
                save_stats_json(training_history, stats_json)
            else:
                fallback_save_json(training_history, stats_json)

    csv_f.close()

    if ENABLE_LOGGING:
        if save_stats_json is not None:
            save_stats_json(training_history, stats_json)
        else:
            fallback_save_json(training_history, stats_json)

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
