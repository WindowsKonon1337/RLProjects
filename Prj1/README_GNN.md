```
Graph Construction:
  Nodes: 49 (7x7 grid)
  Edges: Up, Down, Left, Right + Self-loops
  Adjacency: Normalized (D^-0.5 * A * D^-0.5)

Input Features (per Node):
  [Floor(1) + Wall(1) + Visited(1) + Agent(1) + CoordX(1) + CoordY(1)]  --->  (Batch, N, 6)
       |
       v
  [GCN Layer 1] (Linear + Aggr + ReLU)   --->  (Batch, N, 64)
       |
       v
  [GCN Layer 2] (Linear + Aggr + ReLU)   --->  (Batch, N, 64)
       |
       v
  [GCN Layer 3] (Linear + Aggr + ReLU)   --->  (Batch, N, 64)
       |
       +---------------------------------------------+
       |                                             |
[Readout A: Global Context]                   [Readout B: Local Context]
       |                                             |
   Mean over ALL nodes                        Select Agent's Node
   (1/N * Sum H_i)                            (H_agent)
       |                                             |
       v                                             v
   [Vec: 64]                                     [Vec: 64]
       |                                             |
       +----------------------+----------------------+
                              |
                   FUSION: [64 + 64] = 128
                              |
                   Linear(128 -> 128) + ReLU
                              |
                   Linear(128 -> 4)   [Raw Logits]
                              |
                   * LOGIT_SCALE      [Scaled Logits]
                              |
                   Categorical(logits) -> Action
```

### Why GNN?
1.  **Connectivity Awareness**: The GNN explicitly knows that "Cell A is connected to Cell B". It learns to propagate value along valid paths (edges) and is blocked by lack of edges (if we removed wall edges, though currently we use grid adjacency).
2.  **Permutation Invariance**: The "Global Context" (Mean Pool) doesn't care if the grid is rotated or shifted; it just captures the *distribution* of node states (e.g. "How many unvisited nodes are there?").
3.  **Local + Global**: By combining the *Agent's Node Embedding* (Local) with the *Mean Graph Embedding* (Global), the policy makes decisions based on immediate surroundings while being aware of the overall map completion status.
