# Floor Coverage RL — REINFORCE Agent

A reinforcement learning agent trained with the **REINFORCE** (Monte Carlo Policy Gradient) algorithm to solve a **coverage path planning** problem in a procedurally generated 2D grid world.

---

## 1. Problem Definition

### Task

The agent operates in a bounded 2D grid containing floor cells and obstacle cells. Its goal is to **visit every reachable floor cell** within a limited number of steps.

### Objective

Maximize the fraction of floor cells visited per episode:

$$
\text{coverage} = \frac{|\mathcal{V}|}{|\mathcal{F}|}
$$

where $\mathcal{V}$ is the set of visited cells and $\mathcal{F}$ is the set of all reachable floor cells.

### Environment Dynamics

- The grid is of size $H \times W$ (default $7 \times 7$) with a perimeter wall and randomly placed interior obstacles.
- At each timestep the agent selects one of 4 movement directions. If the target cell is a floor cell, the agent moves there; otherwise it stays in place.
- **Transitions are deterministic** given the action: there is no stochastic noise in the dynamics. Stochasticity enters only through the policy distribution and through random room generation at the start of each episode.

---

## 2. Environment Specification

**Implementation:** `FloorCoverEnv` in [`reinforce_cover.py`](./reinforce_cover.py)

### State Representation

The observation $s_t \in \mathbb{R}^{4HW + 8}$ is a flat vector formed by concatenating four binary grid maps and a local view:

| Component          | Shape        | Description                                      |
| ------------------ | ------------ | ------------------------------------------------ |
| Explored floor map | $H \times W$ | 1 if cell has been seen and is floor             |
| Explored wall map  | $H \times W$ | 1 if cell has been seen and is wall              |
| Visited map        | $H \times W$ | 1 if cell has been visited (trajectory)          |
| Current position   | $H \times W$ | One-hot map of agent position                    |
| Local view         | 8            | `(is_floor, is_visited)` for each of 4 neighbors |

**`observation_space`:** `shape = (4HW + 8,)`, `dtype = float32`, values in $[0, 1]$.

### Action Space

$$
\mathcal{A} = \{0, 1, 2, 3\} \quad \leftrightarrow \quad \{\text{UP},\ \text{DOWN},\ \text{RIGHT},\ \text{LEFT}\}
$$

**`action_space`:** Discrete, $|\mathcal{A}| = 4$.

Invalid actions (moving into a wall) are **masked** by setting their logits to $-10^9$ before softmax, so the policy never wastes probability mass on them.

### Transition Logic

$$
s_{t+1} = \begin{cases}
\text{move to } (r + \Delta r,\ c + \Delta c) & \text{if target cell is floor} \\
\text{stay at } (r, c) & \text{otherwise}
\end{cases}
$$

The room layout is re-sampled at the start of every episode using `build_random_room` (procedural generation with circles, Gaussians, and line obstacles), seeded by `RANDOM_SEED + episode_index` to ensure reproducibility.

### Episode Termination

An episode ends when **either** condition is met:

| Condition                                                        | Type                      |
| ---------------------------------------------------------------- | ------------------------- |
| $\|\mathcal{V}\| \geq \|\mathcal{F}\|$ — all floor cells visited | **Termination** (success) |
| $t \geq T_{\max}$ — step limit reached                           | **Truncation**            |

Default: $T_{\max} = 200$ (`MAX_STEPS_PER_EPISODE` in `config.py`).

### Reward Function

| Event                              | Reward                      |
| ---------------------------------- | --------------------------- |
| Each step taken                    | $r_{\text{step}} = -0.01$   |
| Moving into a wall (no movement)   | $r_{\text{step}}$ only      |
| Visiting a **new** floor cell      | $r_{\text{new}} = +1.0$     |
| Revisiting an already-visited cell | $r_{\text{revisit}} = -0.2$ |

The total step reward is additive:

$$
r_t = r_{\text{step}} + \begin{cases} r_{\text{new}} & \text{if } (r,c) \notin \mathcal{V} \\ r_{\text{revisit}} & \text{if } (r,c) \in \mathcal{V} \end{cases}
$$

All reward parameters are configurable in `config.py`.

---

## 3. Algorithm

**REINFORCE** with a mean-return baseline to reduce gradient variance.

For each episode $k$ with trajectory $\tau^{(k)} = (s_0, a_0, r_0, \ldots, s_T)$:

$$
G_t^{(k)} = \sum_{t'=t}^{T} \gamma^{t'-t} r_{t'}, \qquad b = \frac{1}{N}\sum_{k=1}^{N} G_0^{(k)}
$$

$$
\nabla_\theta J(\theta) \approx \frac{1}{N} \sum_{k=1}^{N} \sum_{t=0}^{T_k} \left(G_t^{(k)} - b\right) \nabla_\theta \log \pi_\theta(a_t^{(k)} \mid s_t^{(k)})
$$

### Policy Networks

| Model           | Description                                                                    |
| --------------- | ------------------------------------------------------------------------------ |
| `MLP`           | 2-layer fully connected: $N_{\text{obs}} \to 128 \to 128 \to 4$                |
| `CNN`           | 3× Conv2d + BN + ReLU, spatial projection, GAP, local branch; fused → 4 logits |
| `GNN`           | 3× GCN layers on grid graph, global pool + agent node embedding → 4 logits     |
| `HEURISTIC_DFS` | Greedy DFS baseline (no learning)                                              |

Set `MODEL_TYPE` in `config.py` to switch.

---

## 4. Reproducibility

### Configuration

All hyperparameters are in [`config.py`](./config.py). Key parameters:

```python
RANDOM_SEED = 42          # master seed
GRID_SIZE   = 7           # H = W
MODEL_TYPE  = "CNN"       # "MLP" | "CNN" | "GNN" | "HEURISTIC_DFS"
NUM_EPISODES = 30000      # training episodes
EVAL_EPISODES = 20        # evaluation episodes
LR          = 1e-4
GAMMA       = 0.99
MAX_STEPS_PER_EPISODE = 200
```

### Training

Run from the **repository root** (`RLProjects/`):

```bash
uv run Prj1/train.py
```

This will:
- Force `TRAIN_MODE=True`, `SAVE_WEIGHTS=True`
- Train for `NUM_EPISODES` episodes
- Save weights to `Prj1/results/policy_weights_<MODEL_TYPE>.pth`
- Save training stats to `Prj1/results/training_stats_<MODEL_TYPE>.json`
- Write logs to `Prj1/results/game_log.log`

### Evaluation

```bash
uv run Prj1/eval.py
```

This will:
- Force `TRAIN_MODE=False`, `LOAD_EXISTING_WEIGHTS=True`
- Load weights from `Prj1/results/policy_weights_<MODEL_TYPE>.pth`
- Run `EVAL_EPISODES` episodes without gradient updates
- Save inference stats to `Prj1/results/inference_stats_<MODEL_TYPE>.json`

> **Note:** Run `eval.py` only after training — it will exit with an error if the weights file is missing.

### DFS Heuristic Baseline

```bash
uv run Prj1/heuristic_run.py
```

Saves results to `Prj1/results/`.

### Visualize Room Graph

```bash
uv run Prj1/visualize_room_graph.py --save Prj1/assets/graph.png
# or
uv run Prj1/visualize_room_graph.py --help
```

---

## 5. Expected Outputs

| File                                        | Description                                    |
| ------------------------------------------- | ---------------------------------------------- |
| `Prj1/results/policy_weights_<MODEL>.pth`   | Saved PyTorch model weights                    |
| `Prj1/results/training_stats_<MODEL>.json`  | Per-episode `{episode, reward, visited_ratio}` |
| `Prj1/results/inference_stats_<MODEL>.json` | Same format, for eval runs                     |
| `Prj1/results/training_plot_<MODEL>.png`    | Learning curve (if `PLOT_TRAINING_CURVE=True`) |
| `Prj1/results/game_log.log`                 | Timestamped log of all training/eval runs      |

---

## 6. Project Structure

```
Prj1/
├── config.py               # All hyperparameters and paths
├── train.py                # Training entry point
├── eval.py                 # Evaluation entry point
├── reinforce_cover.py      # Environment (FloorCoverEnv) + policy networks + training loop
├── room_graph.py           # Procedural room generation
├── heuristic_run.py        # DFS heuristic baseline
├── visualization.py        # Pygame renderer
├── visualization_utils.py  # Plotting and JSON stats utilities
├── visualize_room_graph.py # Room graph visualizer
├── results/                # Training outputs (weights, stats, logs, plots)
├── assets/                 # Images and GIFs for documentation
└── PROJECT_DESCRIPTION.md  # Detailed algorithm description and analysis
```

---

## 7. Dependencies

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) first, then:

```bash
uv sync
```

This installs all dependencies declared in `pyproject.toml` (`torch`, `numpy`, `pygame-ce`, `matplotlib`, etc.) into an isolated virtual environment.

**Activate the environment:**

- **Windows**:
  ```powershell
  .venv\Scripts\activate
  ```
- **Linux/macOS**:
  ```bash
  source .venv/bin/activate
  ```

Python ≥ 3.12 required. Set `ENABLE_VISUALIZATION=False` in `config.py` to run headless (without pygame).


