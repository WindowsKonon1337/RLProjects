"""
train.py — Training script for the Floor Coverage RL agent.

Forces TRAIN_MODE=True and SAVE_WEIGHTS=True regardless of config.py values.
All output is logged via Python's logging module.
"""

import logging
import random
import sys
import os

# ── Override config before importing anything else ──────────────────────────
import config as _cfg
_cfg.TRAIN_MODE = True
_cfg.SAVE_WEIGHTS = True
_cfg.LOAD_EXISTING_WEIGHTS = False   # start fresh; change manually if resuming

# ── Now import the rest (they will see the patched config via `from config import *`) ──
from config import (
    GRID_SIZE, OBSTACLE_PROB, NUM_EPISODES, LR, GAMMA,
    RANDOM_SEED, LOG_FILE, LOG_LEVEL, ENABLE_LOGGING, MODEL_TYPE,
    WEIGHTS_FILE, FPS, AUTO_STEP_DELAY, ENABLE_VISUALIZATION,
)
from room_graph import build_random_room
from reinforce_cover import train_reinforce


def setup_logger() -> logging.Logger:
    logger = logging.getLogger("train")
    logger.setLevel(getattr(logging, LOG_LEVEL.upper(), logging.INFO))
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler (optional)
    if ENABLE_LOGGING and LOG_FILE:
        os.makedirs(os.path.dirname(LOG_FILE) if os.path.dirname(LOG_FILE) else ".", exist_ok=True)
        fh = logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


def main():
    logger = setup_logger()
    logger.info("=== TRAINING MODE ===")
    logger.info(f"Model: {MODEL_TYPE} | Grid: {GRID_SIZE}x{GRID_SIZE} | Episodes: {NUM_EPISODES} | LR: {LR} | Seed: {RANDOM_SEED}")
    logger.info(f"Weights will be saved to: {WEIGHTS_FILE}")
    logger.info(f"Visualization: {ENABLE_VISUALIZATION} | FPS: {FPS} | Step Delay: {AUTO_STEP_DELAY}ms")

    room = build_random_room(
        rows=GRID_SIZE,
        cols=GRID_SIZE,
        add_walls=True,
        max_obstacle_ratio=OBSTACLE_PROB,
        rng=random.Random(RANDOM_SEED),
    )
    logger.info(f"Room built. Floor cells: {len(room.floor_cells())}")

    policy = train_reinforce(
        room,
        num_episodes=NUM_EPISODES,
        lr=LR,
        gamma=GAMMA,
        seed=RANDOM_SEED,
    )
    logger.info("Training complete.")
    return policy


if __name__ == "__main__":
    main()
