import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
from typing import List, Tuple, Optional
from collections import deque

# Import config
from config import *

# Import visualization if needed
try:
    import pygame
    from visualization import init_pygame, draw_grid
except ImportError:
    print("Pygame not found. Visualization disabled.")
    ENABLE_VISUALIZATION = False

# Import plotting utils
try:
    from visualization_utils import plot_training_results, save_stats_json, load_stats_json
except ImportError:
    print("visualization_utils not found or matplotlib missing.")
    PLOT_TRAINING_CURVE = False

from room_graph import (
    RoomGraph,
    FLOOR,
    DIRECTIONS,
    UP,
    DOWN,
    RIGHT,
    LEFT,
    build_random_room,
)


UNKNOWN, EXPLORED_FLOOR, EXPLORED_WALL = 0, 1, 2


class FloorCoverEnv:
    def __init__(
        self,
        room: RoomGraph,
        max_steps: Optional[int] = None,
        reward_new: float = REWARD_NEW,
        reward_revisit: float = REWARD_VISITED,
        reward_step: float = REWARD_STEP,
    ):
        self.room = room
        self.rows, self.cols = room.rows, room.cols
        self.floor_cells = room.floor_cells()
        self.n_floor = len(self.floor_cells)
        self.max_steps = max_steps if max_steps is not None else MAX_STEPS_PER_EPISODE
        self.reward_new = reward_new
        self.reward_revisit = reward_revisit
        self.reward_step = reward_step

        self._pos: Tuple[int, int] = (0, 0)
        self._visited: set = set()
        self._step_count = 0
        self._explored: np.ndarray = np.zeros((self.rows, self.cols), dtype=np.int32)

    def _reveal(self, r: int, c: int) -> None:
        for nr, nc in [(r, c)] + [(r + dr, c + dc) for dr, dc in DIRECTIONS]:
            if 0 <= nr < self.rows and 0 <= nc < self.cols:
                self._explored[nr, nc] = EXPLORED_FLOOR if self.room.is_floor(nr, nc) else EXPLORED_WALL

    def reset(self, start: Optional[Tuple[int, int]] = None) -> Tuple[np.ndarray, dict]:
        if start is not None and self.room.is_floor(start[0], start[1]):
            self._pos = start
        else:
            self._pos = random.choice(self.floor_cells)
        
        self._visited = {self._pos}
        self._step_count = 0
        self._explored.fill(UNKNOWN)
        self._reveal(self._pos[0], self._pos[1])
        return self._get_obs(), self._get_info()

    def _get_local_view(self) -> np.ndarray:
        r, c = self._pos
        local = np.zeros(8, dtype=np.float32)
        for a, (dr, dc) in enumerate(DIRECTIONS):
            nr, nc = r + dr, c + dc
            is_floor = 1.0 if self.room.is_floor(nr, nc) else 0.0
            is_visited = 1.0 if (is_floor and (nr, nc) in self._visited) else 0.0
            local[2 * a] = is_floor
            local[2 * a + 1] = is_visited
        return local

    def _get_obs(self) -> np.ndarray:
        explored_floor = (self._explored == EXPLORED_FLOOR).astype(np.float32)
        explored_wall = (self._explored == EXPLORED_WALL).astype(np.float32)
        
        # New: visited map (binary) for global CNN
        visited_map = np.zeros((self.rows, self.cols), dtype=np.float32)
        for r, c in self._visited:
            visited_map[r, c] = 1.0
            
        current = np.zeros((self.rows, self.cols), dtype=np.float32)
        current[self._pos[0], self._pos[1]] = 1.0
        local = self._get_local_view()
        
        obs = np.concatenate([
            explored_floor.ravel(),
            explored_wall.ravel(),
            visited_map.ravel(), # Added visited map
            current.ravel(),
            local,
        ])
        return obs.astype(np.float32)

    @property
    def obs_size(self) -> int:
        # 4 full grids (explored_floor, explored_wall, visited_map, current_pos) + 8 local vars
        return 4 * self.rows * self.cols + 8

    def _get_info(self) -> dict:
        return {
            "visited": len(self._visited), 
            "n_floor": self.n_floor,
            "agent_pos": self._pos,
            "visited_set": self._visited.copy()
        }

    def get_action_mask(self) -> np.ndarray:
        r, c = self._pos
        mask = np.zeros(4, dtype=np.float32)
        for a, (dr, dc) in enumerate(DIRECTIONS):
            if self.room.is_floor(r + dr, c + dc):
                mask[a] = 1.0
        return mask

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        r, c = self._pos
        dr, dc = DIRECTIONS[action]
        nr, nc = r + dr, c + dc

        reward = self.reward_step

        if not self.room.is_floor(nr, nc):
            # Wall hit: reward remains just the step penalty
            done = False
        else:
            self._pos = (nr, nc)
            self._step_count += 1
            self._reveal(nr, nc)
            if (nr, nc) in self._visited:
                reward += self.reward_revisit
            else:
                self._visited.add((nr, nc))
                reward += self.reward_new
            done = len(self._visited) >= self.n_floor or self._step_count >= self.max_steps

        return self._get_obs(), reward, done, False, self._get_info()



class PolicyNet(nn.Module):
    def __init__(self, obs_size: int, hidden: int = 128):
        super().__init__()
        self.obs_size = obs_size
        self.fc = nn.Sequential(
            nn.Linear(obs_size, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 4),
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        logits = self.fc(x)
        if mask is not None:
            logits = logits.masked_fill(mask <= 0.5, -1e9)
        return logits

    def get_action(self, x: torch.Tensor, mask: torch.Tensor, deterministic: bool = False) -> Tuple[int, torch.Tensor]:
        with torch.no_grad():
            logits = self.forward(x, mask)
            if deterministic:
                action = logits.argmax(dim=-1).item()
            else:
                probs = F.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs)
                action = dist.sample().item()
        logits = self.forward(x, mask)
        log_prob = F.log_softmax(logits, dim=-1)[0, action]
        return action, log_prob

    def save_weights(self, filename: str):
        torch.save(self.state_dict(), filename)
        print(f"Weights saved to {filename}")

    def load_weights(self, filename: str, device: torch.device):
        if os.path.exists(filename):
            self.load_state_dict(torch.load(filename, map_location=device))
            print(f"Weights loaded from {filename}")
        else:
            print(f"Weights file {filename} not found.")


class CNNPolicyNet(nn.Module):
    def __init__(self, rows: int, cols: int, hidden: int = 512):
        super().__init__()
        self.rows = rows
        self.cols = cols
        
        # --- Final Architecture: Parallel Branches + CoordConv ---
        
        # 1. Inputs
        # Global: 4 dynamic channels (Floor, Wall, Visit, Pos) + 2 static (CoordX, CoordY) = 6
        self.input_channels = 6
        
        # 2. Spatial Branch (The Map) -> No Pooling, Stride 1
        # 2. Spatial Branch (The Map) -> No Pooling, Stride 1
        self.conv1 = nn.Conv2d(self.input_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        
        # Projection for Fusion Bottleneck
        # Deep RL trick: Compress massive spatial maps so they don't drown out other signals
        self.flat_spatial_size = 64 * rows * cols
        self.spatial_proj = nn.Linear(self.flat_spatial_size, 256)
        
        # 3. Global Context Branch (The Stats)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.gap_size = 64
        
        # 4. Local Branch (The Safety)
        self.local_fc = nn.Linear(8, 32)
        
        # 5. Fusion & Output
        # Concat: [Spatial_Proj(256), GAP(64), Local(32)] = 352
        fusion_size = 256 + 64 + 32
        self.fc1 = nn.Linear(fusion_size, hidden)
        self.fc2 = nn.Linear(hidden, 4) # Action logits
        
        # Initialization
        self._init_weights()

    def _init_weights(self):
        # Orthogonal Initialization for stability
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size = x.shape[0]
        device = x.device
        
        # Split Global / Local
        global_size = 4 * self.rows * self.cols
        global_part = x[:, :global_size]
        local_part = x[:, global_size:] 
        
        # Reshape Global
        global_view = global_part.view(batch_size, 4, self.rows, self.cols)
        
        # --- ADD CoordConv Channels ---
        # Create normalized meshgrid (0..1)
        xx = torch.linspace(0, 1, self.cols, device=device)
        yy = torch.linspace(0, 1, self.rows, device=device)
        grid_y, grid_x = torch.meshgrid(yy, xx, indexing='ij')
        
        # Expand to batch: (B, 2, R, C)
        coords = torch.stack([grid_y, grid_x], dim=0).unsqueeze(0).expand(batch_size, -1, -1, -1)
        
        # Concat -> (B, 6, R, C)
        x_in = torch.cat([global_view, coords], dim=1)
        
        # --- Spatial Branch ---
        h = F.relu(self.bn1(self.conv1(x_in)))
        h = F.relu(self.bn2(self.conv2(h)))
        h = F.relu(self.bn3(self.conv3(h)))
        
        # Path A: Spatial Projection
        spatial_flat = h.view(batch_size, -1)
        spatial_proj = F.relu(self.spatial_proj(spatial_flat))
        
        # Path B: Global Context
        global_ctx = self.gap(h).view(batch_size, -1)
        
        # --- Local Branch ---
        local_feat = F.relu(self.local_fc(local_part))
        
        # --- Fusion ---
        combined = torch.cat([spatial_proj, global_ctx, local_feat], dim=1)
        
        # Body
        h_fc = F.relu(self.fc1(combined))
        
        # Logit Scaling (Temperature)
        # Scale down to flatten distribution and prevent saturation
        # Uses LOGIT_SCALE from config (e.g., 0.05 for very soft max)
        logits = self.fc2(h_fc) * LOGIT_SCALE
        
        if mask is not None:
            logits = logits.masked_fill(mask <= 0.5, -1e9)
            
        return logits

    def get_action(self, x: torch.Tensor, mask: torch.Tensor, deterministic: bool = False) -> Tuple[int, torch.Tensor]:
        with torch.no_grad():
            logits = self.forward(x, mask)
            if deterministic:
                action = logits.argmax(dim=-1).item()
            else:
                probs = F.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs)
                action = dist.sample().item()
        logits = self.forward(x, mask)
        log_prob = F.log_softmax(logits, dim=-1)[0, action]
        return action, log_prob

    def save_weights(self, filename: str):
        torch.save(self.state_dict(), filename)
        print(f"Weights saved to {filename}")

    def load_weights(self, filename: str, device: torch.device):
        if os.path.exists(filename):
            self.load_state_dict(torch.load(filename, map_location=device))
            print(f"Weights loaded from {filename}")
        else:
            print(f"Weights file {filename} not found.")


# --- GNN Implementation ---

def get_grid_adjacency(rows: int, cols: int, device: torch.device) -> torch.Tensor:
    """
    Builds the adjacency matrix for a grid graph with self-loops.
    Returns normalized adjacency matrix (D^-0.5 * A * D^-0.5).
    """
    num_nodes = rows * cols
    adj = torch.eye(num_nodes, device=device) # Start with self-loops
    
    # Helper to get index
    def get_idx(r, c):
        return r * cols + c
        
    for r in range(rows):
        for c in range(cols):
            curr_idx = get_idx(r, c)
            # Add neighbors
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    neighbor_idx = get_idx(nr, nc)
                    adj[curr_idx, neighbor_idx] = 1.0
                    
    # Normalize: D^-0.5 * A * D^-0.5
    degrees = adj.sum(dim=1)
    d_inv_sqrt = torch.pow(degrees, -0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
    
    norm_adj = d_mat_inv_sqrt @ adj @ d_mat_inv_sqrt
    return norm_adj

class GCNLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)
        
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # x: (Batch, Nodes, Features)
        # adj: (Nodes, Nodes) - assumed shared across batch or (Batch, Nodes, Nodes)
        
        # 1. Linear Transform: X * W
        out = self.linear(x) 
        
        # 2. Message Passing: A * (XW)
        # Check if batch
        if adj.dim() == 2:
            out = torch.matmul(adj, out) # Broadcast adj over batch
        else:
            out = torch.bmm(adj, out)
            
        return out

class GNNPolicyNet(nn.Module):
    def __init__(self, rows: int, cols: int, hidden: int = 64):
        super().__init__()
        self.rows = rows
        self.cols = cols
        self.num_nodes = rows * cols
        
        # Inputs: Floor(1), Wall(1), Visited(1), Agent(1), CoordX(1), CoordY(1) = 6
        self.input_features = 6
        
        # Adjacency matrix (computed lazily or stored buffer)
        self.register_buffer('adj', None)
        
        # GCN Layers
        self.gcn1 = GCNLayer(self.input_features, hidden)
        self.gcn2 = GCNLayer(hidden, hidden)
        self.gcn3 = GCNLayer(hidden, hidden)
        
        # Readout & Policy
        # We concat [GlobalMeanPool, AgentNodeEmbedding]
        self.fc1 = nn.Linear(hidden + hidden, 128)
        self.fc2 = nn.Linear(128, 4)
        
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    
    def _ensure_adj(self, device):
        if self.adj is None:
            self.adj = get_grid_adjacency(self.rows, self.cols, device)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x input is flattened vector from environment: (Batch, obs_size)
        # We need to reshape/process it into (Batch, Nodes, Features)
        
        batch_size = x.shape[0]
        device = x.device
        self._ensure_adj(device)
        
        # 1. Parse Input
        # Obs: [ExploreFloor(N), ExploreWall(N), Visited(N), Agent(N), Local(8)]
        # We ignore Local(8) for the graph part, or append it? 
        # Plan: Use the 4 grid maps + Generate Coords
        
        grid_size = 4 * self.num_nodes
        grid_part = x[:, :grid_size].view(batch_size, 4, self.num_nodes)
        # grid_part shape: (B, 4, N)
        # We want (B, N, 4)
        node_features = grid_part.permute(0, 2, 1) 
        
        # Add Coords
        xx = torch.linspace(0, 1, self.cols, device=device)
        yy = torch.linspace(0, 1, self.rows, device=device)
        grid_y, grid_x = torch.meshgrid(yy, xx, indexing='ij')
        coords = torch.stack([grid_y.flatten(), grid_x.flatten()], dim=1) # (N, 2)
        coords = coords.unsqueeze(0).expand(batch_size, -1, -1) # (B, N, 2)
        
        # Combine -> (B, N, 6)
        h = torch.cat([node_features, coords], dim=-1)
        
        # 2. GCN Layers
        h = F.relu(self.gcn1(h, self.adj))
        h = F.relu(self.gcn2(h, self.adj))
        h = F.relu(self.gcn3(h, self.adj)) # (B, N, hidden)
        
        # 3. Readout
        # A. Global Pooling (Mean)
        global_ctx = h.mean(dim=1) # (B, hidden)
        
        # B. Agent Node Embedding
        # Find agent index from input x (channel 3 is agent pos)
        # agent_map = x[:, 3*N : 4*N] -> argmax
        agent_global_idx = x[:, 3*self.num_nodes : 4*self.num_nodes].argmax(dim=1) # (B,)
        
        # Gather embedding from h using batch index
        # h: (B, N, H)
        agent_embed = h[torch.arange(batch_size), agent_global_idx] # (B, H)
        
        # 4. Fusion
        combined = torch.cat([global_ctx, agent_embed], dim=1) # (B, 2*H)
        
        # 5. Policy Head
        h_fc = F.relu(self.fc1(combined))
        logits = self.fc2(h_fc) * LOGIT_SCALE
        
        if mask is not None:
            logits = logits.masked_fill(mask <= 0.5, -1e9)
            
        return logits
        
    # Standard Policy methods interface
    def get_action(self, x: torch.Tensor, mask: torch.Tensor, deterministic: bool = False) -> Tuple[int, torch.Tensor]:
        with torch.no_grad():
            logits = self.forward(x, mask)
            if deterministic:
                action = logits.argmax(dim=-1).item()
            else:
                probs = F.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs)
                action = dist.sample().item()
        logits = self.forward(x, mask)
        log_prob = F.log_softmax(logits, dim=-1)[0, action]
        return action, log_prob

    def save_weights(self, filename: str):
        torch.save(self.state_dict(), filename)
        print(f"Weights saved to {filename}")

    def load_weights(self, filename: str, device: torch.device):
        if os.path.exists(filename):
            try:
                self.load_state_dict(torch.load(filename, map_location=device), strict=False)
                print(f"Weights loaded from {filename}")
            except Exception as e:
                print(f"Error loading weights: {e}")
        else:
            print(f"Weights file {filename} not found.")


def compute_returns(rewards: List[float], gamma: float = 0.99) -> List[float]:
    R = 0.0
    returns = []
    for r in reversed(rewards):
        R = r + gamma * R
        returns.append(R)
    return list(reversed(returns))


def train_reinforce(
    room: RoomGraph,
    num_episodes: int = NUM_EPISODES,
    lr: float = 1e-3,
    gamma: float = 0.99,
    max_steps: Optional[int] = None,
    seed: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> PolicyNet:
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = FloorCoverEnv(room, max_steps=max_steps)
    
    policy = None
    if MODEL_TYPE == "CNN":
        policy = CNNPolicyNet(room.rows, room.cols).to(device)
        print(f"Initialized CNN Policy (Rows={room.rows}, Cols={room.cols})")
    elif MODEL_TYPE == "GNN":
        policy = GNNPolicyNet(room.rows, room.cols).to(device)
        print(f"Initialized GNN Policy (Nodes={room.rows*room.cols})")
    else:
        policy = PolicyNet(env.obs_size).to(device)
        print(f"Initialized MLP Policy (Input Size={env.obs_size})")

    
    # Load weights if configured
    if LOAD_EXISTING_WEIGHTS:
        policy.load_weights(WEIGHTS_FILE, device)
        
    optimizer = None
    if MODEL_TYPE == "CNN":
        # Regularization on Input Weights (conv1.weight)
        # param_groups allows different weight_decay
        optimizer = torch.optim.Adam([
            {'params': policy.conv1.parameters(), 'weight_decay': CNN_INPUT_REGULARIZATION},
            {'params': policy.conv2.parameters()}, # Default 0
            {'params': policy.conv3.parameters()},
            {'params': policy.bn1.parameters()},
            {'params': policy.bn2.parameters()},
            {'params': policy.bn3.parameters()},
            {'params': policy.local_fc.parameters()},
            {'params': policy.fc1.parameters()},
            {'params': policy.fc2.parameters()}
        ], lr=lr)
    else:
        optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

    # Initialize Pygame if enabled
    screen = None
    font = None
    small_font = None
    tiny_font = None
    clock = None
    
    if ENABLE_VISUALIZATION:
        screen, font, small_font, tiny_font, clock = init_pygame()
        
    # Persistent auto_mode state
    auto_mode = False 

    # Initialize stats
    training_history: List[dict] = []
    start_episode = 0
    
    if LOAD_EXISTING_WEIGHTS:
        # Load previous history if weights are loaded
        stats_file = TRAINING_STATS_FILE if TRAIN_MODE else INFERENCE_STATS_FILE
        training_history = load_stats_json(stats_file)
        if training_history:
            start_episode = training_history[-1]['episode']
            print(f"Resuming history from {stats_file}, episode {start_episode}")

    try:
        for episode_idx in range(num_episodes):
            episode = start_episode + episode_idx + 1 # Cumulative episode number
            
            # Generate new room for each episode
            # We use the dimensions of the passed 'room' to be safe, or from config
            current_room = build_random_room(
                rows=room.rows, 
                cols=room.cols, 
                add_walls=True, 
                max_obstacle_ratio=OBSTACLE_PROB, 
                rng=random.Random() # New random seed each time (or omitted to use system time)
            )
            
            # Re-initialize environment with the new room
            # Note: We assume obs_size remains constant (based on fixed rows/cols)
            env = FloorCoverEnv(current_room, max_steps=max_steps)

            obs, _ = env.reset() # Random start
            traj_log_probs: List[torch.Tensor] = []
            rewards: List[float] = []
            
            step = 0
            G = 0.0
            reset_requested = False

            while True:
                step_once = False
                
                # User Action Override
                manual_action = None
                
                # Handle Pygame events if visualization is enabled
                if ENABLE_VISUALIZATION:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            pygame.quit()
                            sys.exit()
                        elif event.type == pygame.KEYDOWN:
                            if event.key == pygame.K_ESCAPE:
                                pygame.quit()
                                sys.exit()
                            if event.key == pygame.K_SPACE:
                                if auto_mode:
                                    auto_mode = False # Stop auto if running
                                else:
                                    step_once = True # Single step if paused
                            if event.key == pygame.K_a:
                                auto_mode = not auto_mode # Toggle auto
                            if event.key == pygame.K_r: # Reset Key (User req 3)
                                reset_requested = True
                                
                            # Manual Control (Arrows)
                            # Remapped for Visual Orientation (Row=X, Col=Y flipped)
                            if event.key == pygame.K_UP:
                                manual_action = RIGHT # Visual Up
                            elif event.key == pygame.K_DOWN:
                                manual_action = LEFT  # Visual Down
                            elif event.key == pygame.K_LEFT:
                                manual_action = UP    # Visual Left
                            elif event.key == pygame.K_RIGHT:
                                manual_action = DOWN  # Visual Right
                
                if reset_requested:
                    break # Break inner loop, will start next episode

                # Decide if we should proceed with a step
                # Step if auto_mode, OR step_once, OR manual action was pressed
                should_step = auto_mode or step_once or (manual_action is not None)

                # If NOT stepping (Paused), just draw and wait
                if not should_step and ENABLE_VISUALIZATION:
                    # Need to construct state for draw_grid
                    state_info = env._get_info()
                    vis_state = {
                        'agent_pos': state_info['agent_pos'],
                        'visited': state_info['visited_set'],
                    }
                    
                    # Get action probs for visualization (optional)
                    x = torch.from_numpy(obs).float().unsqueeze(0).to(device)
                    mask = torch.from_numpy(env.get_action_mask()).float().unsqueeze(0).to(device)
                    with torch.no_grad():
                        logits = policy(x, mask)
                        probs = F.softmax(logits, dim=-1).cpu().numpy()[0]
                    
                    # waiting=True shows "WAITING"
                    draw_grid(screen, font, small_font, tiny_font, env.room, vis_state, G, episode, True, step, action_probs=probs)
                    clock.tick(FPS)
                    continue

                # Taking a step
                x = torch.from_numpy(obs).float().unsqueeze(0).to(device)  # (1, obs_size)
                mask = torch.from_numpy(env.get_action_mask()).float().unsqueeze(0).to(device)
                
                if manual_action is not None:
                     # Force manual action (User Request)
                     # Even if it's a wall (mask=0), we perform the action.
                     # Env will handle the wall collision (no move, penalty).
                     action = manual_action

                     # Compute log_prob for the manual action.
                     # CRITICAL: We must NOT pass the mask to policy() here, because if the action 
                     # is a wall, masked logits would be -1e9, leading to log_prob = -inf.
                     # We want the "raw" probability the policy assigns to this action.
                     logits = policy(x, mask=None) 
                     log_prob = F.log_softmax(logits, dim=-1)[0, action]
                else:
                    action, log_prob = policy.get_action(x, mask, deterministic=False)
                
                traj_log_probs.append(log_prob)
                
                # Calculate action probs for visualization
                probs = None
                if ENABLE_VISUALIZATION:
                     with torch.no_grad():
                        logits = policy(x, mask)
                        probs = F.softmax(logits, dim=-1).cpu().numpy()[0]

                obs, reward, done, _, info = env.step(action)
                rewards.append(reward)
                G += reward
                step += 1
                
                if ENABLE_VISUALIZATION:
                    state_info = info
                    vis_state = {
                        'agent_pos': state_info['agent_pos'],
                        'visited': state_info['visited_set'],
                    }
                    if episode < HEADLESS_EPISODES:
                         pygame.event.pump() # Keep window responsive
                    else:
                        # waiting=False shows "Episode X Step Y"
                        draw_grid(screen, font, small_font, tiny_font, env.room, vis_state, G, episode, False, step, action_probs=probs)
                        pygame.display.flip()
                    if auto_mode and AUTO_STEP_DELAY > 0:
                        pygame.time.delay(AUTO_STEP_DELAY)

                if done:
                    break
            
            # Policy Update
            if TRAIN_MODE:
                returns = compute_returns(rewards, gamma=gamma)
                baseline = sum(returns) / len(returns) if returns else 0.0
                policy_loss = 0.0
                for log_prob, G_ret in zip(traj_log_probs, returns):
                    policy_loss = policy_loss - log_prob * (G_ret - baseline)
                optimizer.zero_grad()
                policy_loss.backward()
                optimizer.step()
            
            # --- Stats Collection ---
            n_visited = env._get_info()["visited"]
            total_reward = sum(rewards)
            visited_ratio = n_visited / env.n_floor
            
            training_history.append({
                'episode': episode,
                'reward': total_reward,
                'visited_ratio': visited_ratio
            })

            # Calculate moving average of reward (last 100 episodes)
            # We can use training_history to get recent rewards efficiently
            recent_rewards = [h['reward'] for h in training_history[-100:]]
            avg_reward = sum(recent_rewards) / len(recent_rewards) if recent_rewards else 0.0

            if episode % 100 == 0 or episode_idx == 0:
                print(f"Episode {episode}: visited {n_visited}/{env.n_floor}, avg_reward_100={avg_reward:.2f}")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    finally:
        if SAVE_WEIGHTS and TRAIN_MODE:
            policy.save_weights(WEIGHTS_FILE)
            
        # Save persistence stats
        if training_history:
            stats_file = TRAINING_STATS_FILE if TRAIN_MODE else INFERENCE_STATS_FILE
            save_stats_json(training_history, stats_file)
            if PLOT_TRAINING_CURVE and TRAIN_MODE:
                plot_training_results(training_history, TRAINING_PLOT_FILE)

    return policy



def evaluate_policy(room: RoomGraph, policy: PolicyNet, num_episodes: int = 5, device: Optional[torch.device] = None) -> None:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    policy.eval()
    
    # Check if we should visualize evaluation
    screen = None
    if ENABLE_VISUALIZATION:
         screen, font, small_font, tiny_font, clock = init_pygame()
         
    for ep in range(num_episodes):
        # Generate new room for evaluation as well
        current_room = build_random_room(
            rows=room.rows, 
            cols=room.cols, 
            add_walls=True, 
            max_obstacle_ratio=OBSTACLE_PROB,
            rng=random.Random() 
        )
        
        env = FloorCoverEnv(current_room)
        
        obs, _ = env.reset()
        steps = 0
        G = 0.0
        while steps < env.max_steps:
             # Handle Pygame events
            if ENABLE_VISUALIZATION:
                for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            pygame.quit()
                            sys.exit()
                        if event.key == pygame.K_ESCAPE:
                            pygame.quit()
                            sys.exit()

            x = torch.from_numpy(obs).float().unsqueeze(0).to(device)
            mask = torch.from_numpy(env.get_action_mask()).float().unsqueeze(0).to(device)
            action, _ = policy.get_action(x, mask, deterministic=True)
            
            # Vis info
            probs = None
            if ENABLE_VISUALIZATION:
                with torch.no_grad():
                    logits = policy(x, mask)
                    probs = F.softmax(logits, dim=-1).cpu().numpy()[0]
                    
            obs, reward, done, _, info = env.step(action)
            steps += 1
            G += reward
            
            if ENABLE_VISUALIZATION:
                state_info = info
                vis_state = {
                    'agent_pos': state_info['agent_pos'],
                    'visited': state_info['visited_set'],
                    'neighbors': []
                }
                draw_grid(screen, font, small_font, tiny_font, env.room, vis_state, G, ep, False, steps, action_probs=probs)
                pygame.time.delay(AUTO_STEP_DELAY)

            if done:
                break
        print(f"  Eval ep {ep + 1}: visited {info['visited']}/{env.n_floor} in {steps} steps")


def main():
    room = build_random_room(rows=GRID_SIZE, cols=GRID_SIZE, add_walls=True, max_obstacle_ratio=OBSTACLE_PROB, rng=random.Random(42))
    print("Floor cells:", len(room.floor_cells()))
    policy = train_reinforce(room, num_episodes=NUM_EPISODES, lr=LR, gamma=GAMMA, seed=42)
    print("Evaluation (deterministic):")
    evaluate_policy(room, policy, num_episodes=5)


if __name__ == "__main__":
    main()
