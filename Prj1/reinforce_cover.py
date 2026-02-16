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
            if self.room.is_floor(START_POSITION[0], START_POSITION[1]):
                self._pos = START_POSITION
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
        current = np.zeros((self.rows, self.cols), dtype=np.float32)
        current[self._pos[0], self._pos[1]] = 1.0
        local = self._get_local_view()
        obs = np.concatenate([
            explored_floor.ravel(),
            explored_wall.ravel(),
            current.ravel(),
            local,
        ])
        return obs.astype(np.float32)

    @property
    def obs_size(self) -> int:
        return 3 * self.rows * self.cols + 8

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


def compute_returns(rewards: List[float], gamma: float = 0.99) -> List[float]:
    R = 0.0
    returns = []
    for r in reversed(rewards):
        R = r + gamma * R
        returns.append(R)
    return list(reversed(returns))


def train_reinforce(
    room: RoomGraph,
    num_episodes: int = 2000,
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
    policy = PolicyNet(env.obs_size).to(device)
    
    # Load weights if configured
    if LOAD_EXISTING_WEIGHTS:
        policy.load_weights(WEIGHTS_FILE, device)
        
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

    # Force start position to be a floor cell if possible (User req 1)
    # We do this once on the passed room
    sx, sy = START_POSITION
    if 0 <= sx < room.rows and 0 <= sy < room.cols:
        room.set_floor(sx, sy)

    for episode in range(num_episodes):
        obs, _ = env.reset(start=START_POSITION) # Always try START_POSITION
        traj_log_probs: List[torch.Tensor] = []
        rewards: List[float] = []
        
        step = 0
        G = 0.0
        reset_requested = False

        while True:
            step_once = False
            
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
            
            if reset_requested:
                break # Break inner loop, will start next episode

            # Decide if we should proceed with a step
            should_step = auto_mode or step_once

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
                # waiting=False shows "Episode X Step Y"
                draw_grid(screen, font, small_font, tiny_font, env.room, vis_state, G, episode, False, step, action_probs=probs)
                
                if episode < HEADLESS_EPISODES:
                     pass
                     
                pygame.display.flip()
                if auto_mode and AUTO_STEP_DELAY > 0:
                    pygame.time.delay(AUTO_STEP_DELAY)

            if done:
                break
        
        # Policy Update
        returns = compute_returns(rewards, gamma=gamma)
        baseline = sum(returns) / len(returns) if returns else 0.0
        policy_loss = 0.0
        for log_prob, G_ret in zip(traj_log_probs, returns):
            policy_loss = policy_loss - log_prob * (G_ret - baseline)
        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()

        if (episode + 1) % 100 == 0 or episode == 0:
            n_visited = env._get_info()["visited"]
            total_reward = sum(rewards)
            print(f"Episode {episode + 1}: visited {n_visited}/{env.n_floor}, total_reward={total_reward:.2f}")

    if SAVE_WEIGHTS:
        policy.save_weights(WEIGHTS_FILE)

    return policy



def evaluate_policy(room: RoomGraph, policy: PolicyNet, num_episodes: int = 5, device: Optional[torch.device] = None) -> None:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = FloorCoverEnv(room)
    policy.eval()
    
    # Check if we should visualize evaluation
    screen = None
    if ENABLE_VISUALIZATION:
         screen, font, small_font, tiny_font, clock = init_pygame()
         
    for ep in range(num_episodes):
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
    policy = train_reinforce(room, num_episodes=2000, lr=1e-3, gamma=0.99, seed=42)
    print("Evaluation (deterministic):")
    evaluate_policy(room, policy, num_episodes=5)


if __name__ == "__main__":
    main()
