"""
eval.py — Evaluation script for the Floor Coverage RL agent.

Forces TRAIN_MODE=False and LOAD_EXISTING_WEIGHTS=True regardless of config.py values.
All output is logged via Python's logging module.
"""

import logging
import random
import sys
import os

# ── Override config before importing anything else ──────────────────────────
import config as _cfg
_cfg.TRAIN_MODE = False
_cfg.LOAD_EXISTING_WEIGHTS = True
_cfg.SAVE_WEIGHTS = False            # never overwrite weights during eval

# ── Now import the rest ──────────────────────────────────────────────────────
from config import (
    GRID_SIZE, OBSTACLE_PROB, EVAL_EPISODES, LR, GAMMA,
    RANDOM_SEED, LOG_FILE, LOG_LEVEL, ENABLE_LOGGING,
    WEIGHTS_FILE, MODEL_TYPE, FPS, AUTO_STEP_DELAY, ENABLE_VISUALIZATION,
)
from room_graph import build_random_room
from reinforce_cover import train_reinforce   # reuses the same loop in inference mode


def setup_logger() -> logging.Logger:
    logger = logging.getLogger("eval")
    logger.setLevel(getattr(logging, LOG_LEVEL.upper(), logging.INFO))
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    if ENABLE_LOGGING and LOG_FILE:
        os.makedirs(os.path.dirname(LOG_FILE) if os.path.dirname(LOG_FILE) else ".", exist_ok=True)
        fh = logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


def main():
    logger = setup_logger()
    logger.info("=== EVALUATION MODE ===")
    logger.info(f"Model: {MODEL_TYPE} | Weights: {WEIGHTS_FILE} | Episodes: {EVAL_EPISODES} | Seed: {RANDOM_SEED}")
    logger.info(f"Visualization: {ENABLE_VISUALIZATION} | FPS: {FPS} | Step Delay: {AUTO_STEP_DELAY}ms")

    if not os.path.exists(WEIGHTS_FILE):
        logger.error(f"Weights file not found: {WEIGHTS_FILE}. Run train.py first.")
        sys.exit(1)

    room = build_random_room(
        rows=GRID_SIZE,
        cols=GRID_SIZE,
        add_walls=True,
        max_obstacle_ratio=OBSTACLE_PROB,
        rng=random.Random(RANDOM_SEED),
    )
    logger.info(f"Room built. Floor cells: {len(room.floor_cells())}")

    # Run inference episodes via the same train_reinforce loop (TRAIN_MODE=False disables updates)
    train_reinforce(
        room,
        num_episodes=EVAL_EPISODES,
        lr=LR,
        gamma=GAMMA,
        seed=RANDOM_SEED,
    )
    logger.info("Evaluation complete.")


if __name__ == "__main__":
    main()
