#!/usr/bin/env python3

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import numpy as np

from Prj1.room_graph import RoomGraph, FLOOR, OBSTACLE, build_random_room

CMAP_ROOM = mcolors.ListedColormap(["#2ecc71", "#e74c3c"])


def visualize(
    room: RoomGraph,
    *,
    show_edges: bool = True,
    cell_size: float = 0.9,
    title: str | None = "Граф комнаты",
) -> plt.Figure:
    rows, cols = room.rows, room.cols
    grid = room.grid

    fig, ax = plt.subplots(1, 1, figsize=(max(6, cols * 0.5), max(5, rows * 0.5)))
    ax.imshow(
        grid,
        cmap=CMAP_ROOM,
        vmin=0,
        vmax=1,
        extent=(0, cols, rows, 0),
        interpolation="nearest",
        aspect="equal",
    )
    ax.set_xticks(np.arange(0.5, cols, 1))
    ax.set_xticklabels(range(cols))
    ax.set_yticks(np.arange(0.5, rows, 1))
    ax.set_yticklabels(range(rows))
    ax.set_xlabel("col")
    ax.set_ylabel("row")
    if title:
        ax.set_title(title)

    patch_floor = mpatches.Patch(color="green", label="Floor")
    patch_obst = mpatches.Patch(color="red", label="Obstacle")
    ax.legend(handles=[patch_floor, patch_obst], loc="upper left")

    if show_edges:
        for r in range(rows):
            for c in range(cols):
                if grid[r, c] != FLOOR:
                    continue
                for nr, nc in room.get_neighbors(r, c):
                    y0, x0 = r + 0.5, c + 0.5
                    y1, x1 = nr + 0.5, nc + 0.5
                    dx, dy = x1 - x0, y1 - y0
                    ax.arrow(
                        x0, y0,
                        dx * cell_size * 0.4,
                        dy * cell_size * 0.4,
                        head_width=0.2,
                        head_length=0.15,
                        fc="navy",
                        ec="navy",
                        alpha=0.6,
                    )
    plt.tight_layout()
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description="Room graph visualization")
    parser.add_argument("--no-edges", action="store_true", help="Do not draw edges (cells only)")
    parser.add_argument("--save", type=str, metavar="FILE", help="Save figure to file")
    parser.add_argument("--no-walls", action="store_true", help="No outer walls")
    parser.add_argument("--example", action="store_true", help="Old example (walls + one rectangle)")
    parser.add_argument("--seed", type=int, default=None, metavar="N", help="Seed for random generation")
    parser.add_argument("--rows", type=int, default=12, help="Grid height")
    parser.add_argument("--cols", type=int, default=16, help="Grid width")
    parser.add_argument("--max-obstacle-ratio", type=float, default=0.3, metavar="R",
                        help="Max obstacle ratio (0..1), default 0.3")
    args = parser.parse_args()

    rng = random.Random(args.seed) if args.seed is not None else None
    room = build_random_room(
        rows=args.rows,
        cols=args.cols,
        add_walls=not args.no_walls,
        max_obstacle_ratio=args.max_obstacle_ratio,
        rng=rng,
    )
    title = "Room graph (circles, gaussians, lines)"
    fig = visualize(
        room,
        show_edges=not args.no_edges,
        title=title,
    )
    if args.save:
        fig.savefig(args.save, dpi=150, bbox_inches="tight")
        print(f"Saved: {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
