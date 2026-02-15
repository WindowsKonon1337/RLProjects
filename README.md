## Installation

Install dependencies with uv:

```bash
uv sync
```

How to install uv: [link](https://docs.astral.sh/uv/getting-started/installation/)

## Running the Simulation

Run the grid world RL game with visualization:

```bash
python grid_game_pygame.py
```

## Headless Mode (No Visualization)

To run without visualization (faster for training):

1. Set `ENABLE_VISUALIZATION = False` in `config.py`
2. Run: `python grid_game_pygame.py`

The game will automatically run episodes and log results to `game_log.log`.

## RL Components


* **State ($s \in \mathcal{S}$):** Position, Left, Upper, Right, Bottom $(x, y)$, $(x-1, y)$, $(x, y+1)$, $(x+1, y)$, $(x, y-1)$. Number of steps $n \in \mathbb{N}$.
* **Action ($a \in \mathcal{A}$):** Input: $s \in \mathcal{S} \to$ Output: $a \in \{0, 1, 2, 3\}$, chosen by policy.
* **Reward ($r$):** Input: $s, a \to$ Output: $r \in \mathbb{R}$ ($1.0$ for new one, $-0.5$ for visited cell, $100.0$ for completion).
* **Policy ($\pi$):** Input: $s \in \mathcal{S} \to$ Output: $\pi(a|s) \in [0, 1]^4$, where $\sum \pi = 1$.


