import sys
import os
import torch
import torch.nn as nn
import reinforce_cover
from room_graph import build_random_room
import config

# Override config to disable visualization for test
reinforce_cover.ENABLE_VISUALIZATION = False
config.ENABLE_VISUALIZATION = False
# Set model type to GNN
reinforce_cover.MODEL_TYPE = "GNN"

def test_gnn_implementation():
    print("Testing GNN Architecture (GCN, Adjacency, Readout)...")
    
    # 1. Initialize Room
    rows, cols = 7, 7
    room = build_random_room(rows=rows, cols=cols, add_walls=True, max_obstacle_ratio=0.1)
    
    # 2. Test Adjacency Generation
    print("Testing Adjacency Matrix Helper...")
    try:
        device = torch.device("cpu")
        adj = reinforce_cover.get_grid_adjacency(rows, cols, device)
        print(f"Adjacency Matrix Shape: {adj.shape}")
        assert adj.shape == (rows*cols, rows*cols)
        # Check self loop (diagonal should be non-zero)
        diag_sum = torch.diag(adj).sum().item()
        print(f"Diagonal Sum (should be > 0): {diag_sum}")
        assert diag_sum > 0
    except Exception as e:
        print(f"FAILED Adjacency Test: {e}")
        import traceback
        traceback.print_exc()
        return

    # 3. Initialize Policy
    print("Initializing GNN Policy...")
    try:
        policy = reinforce_cover.GNNPolicyNet(rows, cols)
        print("Policy Initialized.")
        print(f"  Nodes: {policy.num_nodes}")
        print(f"  Input Features: {policy.input_features}")
        print(f"  GCN Layer 1: {policy.gcn1}")
    except Exception as e:
        print(f"FAILED to initialize policy: {e}")
        import traceback
        traceback.print_exc()
        return

    # 4. Test Forward Pass with dummy data
    print("Testing Forward Pass...")
    # Obs size: 4 * 49 + 8 (local) = 196 + 8 = 204
    dummy_obs = torch.zeros((2, 4 * rows * cols + 8)) # Batch size 2
    # Set agent pos to 0 for batch 0 and 1 for batch 1 to test embedding lookup
    # Channel 3 is agent pos. Index range [3*N, 4*N)
    N = rows * cols
    dummy_obs[0, 3*N + 0] = 1.0 # Agent at node 0
    dummy_obs[1, 3*N + 1] = 1.0 # Agent at node 1
    
    try:
        logits = policy(dummy_obs)
        print(f"Forward pass successful. Output shape: {logits.shape}")
        print(f"Logits range: {logits.min().item():.4f} to {logits.max().item():.4f}")
        assert logits.shape == (2, 4)
    except Exception as e:
        print(f"FAILED forward pass: {e}")
        import traceback
        traceback.print_exc()
        return

    # 5. Test Training Loop (Short)
    print("Running Short Training Loop (2 episodes)...")
    try:
        reinforce_cover.train_reinforce(room, num_episodes=2, lr=1e-3, gamma=0.99, seed=42)
        print("Training loop finished successfully.")
    except Exception as e:
        print(f"FAILED training loop: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_gnn_implementation()
