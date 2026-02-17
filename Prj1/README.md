# Start

launch:

```bash
uv sync
```

How to install uv: [link](https://docs.astral.sh/uv/getting-started/installation/)

How to run: 
```bash
python reinforce_cover.py
```

Change `Config.py` to change parameters.


# Model Architecture

The current policy uses a **Multi-Branch CNN Architecture** (`CNNPolicyNet`).

## Inputs
1. **Global Grid (CNN Branch)**:
   - Shape: `(Batch, 4, Rows, Cols)`
   - Channels:
     1. Explored Floor Map
     2. Explored Wall Map
     3. Visited Cells Map
     4. Current Agent Position
2. **Local View (MLP Branch)**:
   - Shape: `(Batch, 8)`
   - Contains the state of the 8 immediate neighbors.

## Architecture
- **Global Branch**:
  - `Conv2d(4 -> 16, 3x3)` + ReLU
  - `Conv2d(16 -> 32, 3x3)` + ReLU
  - Flatten to vector `32 * Rows * Cols`
- **Local Branch**:
  - `Linear(8 -> 16)` + ReLU
- **Fusion**:
  - Concatenation of Global + Local vectors.
  - `Linear(Combined -> 128)` + ReLU
  - `Linear(128 -> 4)` (Action Logits)

## Configuration
- Model Type: `CNN` (set in `config.py`)
- Hidden Size: 128
- Input Regularization (L2): `1e-4` on the first Conv layer to encourage sparsity.
