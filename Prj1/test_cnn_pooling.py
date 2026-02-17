import sys
import os
import torch
import reinforce_cover
from room_graph import build_random_room
import config

# Override config to disable visualization for test
reinforce_cover.ENABLE_VISUALIZATION = False
config.ENABLE_VISUALIZATION = False

def test_cnn_implementation():
    print("Testing CNN Implementation (Pooling + Scaling)...")
    
    # 1. Initialize Room
    rows, cols = 7, 7
    room = build_random_room(rows=rows, cols=cols, add_walls=True, max_obstacle_ratio=0.1)
    
    # 2. Initialize Policy
    print("Initializing Policy...")
    try:
        policy = reinforce_cover.CNNPolicyNet(rows, cols)
        print("Policy Initialized.")
        print(f"  Max Pool Layer: {policy.pool}")
        print(f"  GAP Layer: {policy.gap}")
        print(f"  Flat Spatial Size: {policy.flat_spatial_size}")
        print(f"  GAP Size: {policy.gap_size}")
    except Exception as e:
        print(f"FAILED to initialize policy: {e}")
        import traceback
        traceback.print_exc()
        return

    # 3. Test Forward Pass with dummy data
    print("Testing Forward Pass...")
    dummy_obs = torch.zeros((1, 4 * rows * cols + 8)) # Global + Local
    try:
        logits = policy(dummy_obs)
        print(f"Forward pass successful. Output shape: {logits.shape}")
        print(f"Logits range (should be small due to scaling): {logits.min().item():.4f} to {logits.max().item():.4f}")
        assert logits.shape == (1, 4)
    except Exception as e:
        print(f"FAILED forward pass: {e}")
        import traceback
        traceback.print_exc()
        return

    # 4. Test Training Loop (Short)
    print("Running Short Training Loop (2 episodes)...")
    try:
        reinforce_cover.train_reinforce(room, num_episodes=2, lr=1e-3, gamma=0.99, seed=42)
        print("Training loop finished successfully.")
    except Exception as e:
        print(f"FAILED training loop: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_cnn_implementation()
