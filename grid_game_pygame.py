"""
Grid World RL Game - Main Module
Interactive reinforcement learning environment with pygame visualization.

Controls (when visualization enabled):
- SPACE: Take one step
- A: Toggle auto mode
- R: Reset game
- ESC/Q: Quit
"""

import sys
import time
from datetime import datetime
from config import (START_POSITION, MAX_STEPS_PER_EPISODE, FPS, AUTO_STEP_DELAY,
                    ENABLE_VISUALIZATION, ENABLE_LOGGING, LOG_FILE, VERBOSE_CONSOLE,
                    REWARD_COMPLETION, HEADLESS_EPISODES)
from scene import create_scene_graph, reset_graph, check_completion, get_completion_stats
from rl_functions import get_state, sample_step, reward
from reinforce import reinforce_update



if ENABLE_VISUALIZATION:
    import pygame
    from visualization import init_pygame, draw_grid

RETURNS_LOG_PATH = "returns.txt"

def init_returns_log(path=RETURNS_LOG_PATH):
    """
    Create (overwrite) returns log file at the start of each run.
    """
    with open(path, "w", encoding="utf-8") as f:
        f.write("episode\tsteps\treturn_G\tcompleted\n")

def append_return_log(episode, steps, G, completed, path=RETURNS_LOG_PATH):
    """
    Append one episode result to returns log.
    """
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"{episode}\t{steps}\t{float(G)}\t{int(completed)}\n")

class GameLogger:
    """Handles logging to both file and console"""
    
    def __init__(self, log_file=LOG_FILE, enable_logging=ENABLE_LOGGING):
        self.enable_logging = enable_logging
        self.log_file = log_file
        
        if self.enable_logging:
            # Clear/create log file
            with open(self.log_file, 'w', encoding='utf-8') as f:
                f.write(f"=== Grid World RL Game Log ===\n")
                f.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    def log(self, message, console=True):
        """Log message to file and optionally console"""
        timestamp = datetime.now().strftime('%H:%M:%S')
        formatted = f"[{timestamp}] {message}"
        
        if console and VERBOSE_CONSOLE:
            print(message)
        
        if self.enable_logging:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(formatted + '\n')
    
    def log_episode_start(self, episode, start_pos):
        """Log episode start"""
        msg = f"\n{'='*60}\nEpisode {episode} Started - Agent at {start_pos}\n{'='*60}"
        self.log(msg)
    
    def log_step(self, step, old_pos, new_pos, step_reward, cumulative_G):
        """Log individual step"""
        moved = "✓" if old_pos != new_pos else "✗ (blocked)"
        msg = f"  Step {step:3d}: {old_pos} → {new_pos} {moved} | r={step_reward:+.1f} | G={cumulative_G:+.1f}"
        self.log(msg, console=False)  # Don't spam console with every step
    
    def log_episode_end(self, episode, steps, G, reason="completed"):
        """Log episode end with statistics"""
        msg = f"\nEpisode {episode} Ended ({reason})\n  Steps: {steps}\n  Total Reward (G): {G:.2f}\n  Avg Reward/Step: {G/max(steps,1):.3f}"
        self.log(msg)
    
    def log_statistics(self, episodes_data):
        """Log overall statistics"""
        if not episodes_data:
            return
        
        total_episodes = len(episodes_data)
        completed_episodes = sum(1 for ep in episodes_data if ep.get('completed', False))
        avg_reward = sum(ep['G'] for ep in episodes_data) / total_episodes
        avg_steps = sum(ep['steps'] for ep in episodes_data) / total_episodes
        
        msg = f"\n{'='*60}\nOVERALL STATISTICS ({total_episodes} episodes)\n{'='*60}"
        msg += f"\n  Completed: {completed_episodes}/{total_episodes} ({completed_episodes/total_episodes*100:.1f}%)"
        msg += f"\n  Average Reward: {avg_reward:.2f}"
        msg += f"\n  Average Steps: {avg_steps:.1f}"
        msg += f"\n  Best Reward: {max(ep['G'] for ep in episodes_data):.2f}"
        msg += f"\n  Worst Reward: {min(ep['G'] for ep in episodes_data):.2f}"
        
        if completed_episodes > 0:
            completed_rewards = [ep['G'] for ep in episodes_data if ep.get('completed', False)]
            msg += f"\n  Avg Reward (completed only): {sum(completed_rewards)/len(completed_rewards):.2f}"
        
        self.log(msg)


logger = GameLogger()
init_returns_log()

def reset_game():
    """
    Reset the game to initial state.
    
    Returns:
        tuple: (graph, state, G, episode, step, waiting)
    """
    graph = create_scene_graph()
    reset_graph(graph)
    
    # Always start at START_POSITION (top-left corner)
    state = get_state(graph, START_POSITION, steps=0)
    G = 0.0
    episode = 1
    step = 0
    waiting = False
    
    return graph, state, G, episode, step, waiting


def do_step(graph, state, G, episode, step, waiting, episodes_data, trajectory):
    """
    Execute one game step.
    
    Args:
        graph: Scene graph
        state: Current state
        G: Cumulative reward
        episode: Episode number
        step: Step count
        waiting: Waiting flag
        episodes_data: List to store episode statistics
        
    Returns:
        tuple: Updated (graph, state, G, episode, step, waiting)
    """
    if waiting:
        # Start new episode
        trajectory.clear()
        graph, state, G, _, step, waiting = reset_game()
        episode += 1
        logger.log_episode_start(episode, START_POSITION)
        return graph, state, G, episode, step, waiting
    
    # Save old position for logging
    old_pos = state['agent_pos']
    
    # Calculate reward before stepping
    #step_reward = reward(graph, state)
    #G += step_reward
    
    # Take step
    state_next, action_idx, probs = sample_step(graph, state)
    step_reward = reward(graph, state_next)
    G += step_reward
    new_pos = state_next['agent_pos']
    step = state_next['steps']

    trajectory.append({
    "state": state,          # state BEFORE action
    "action": action_idx,    # 0..3
    "reward": step_reward,   # reward
    "probs": probs           # pi(.|s)
    })
    state = state_next
    
    # Log step
    logger.log_step(step, old_pos, new_pos, step_reward, G)
    
    # Check if all free cells are visited (COMPLETION)
    if check_completion(graph):
        G += REWARD_COMPLETION  # Add completion bonus
        waiting = True
        stats = get_completion_stats(graph)
        episodes_data.append({'episode': episode, 'steps': step, 'G': G, 'completed': True})
        logger.log(f"\n🎉 COMPLETION! All {stats['total_free']} free cells visited!")
        logger.log_episode_end(episode, step, G, f"COMPLETED (+{REWARD_COMPLETION} bonus)")
        print(f"[DEBUG] episode end, len(traj)={len(trajectory)}")
        _, G_ep = reinforce_update(trajectory, alpha=0.005, gamma=1.0)
        print(f"[DEBUG] reinforce called, G_ep={G_ep}")
        append_return_log(
            episode=episode,
            steps=step,
            G=G,
            completed=True
        )
        return graph, state, G, episode, step, waiting
    
    # Check if episode should end due to max steps
    if step >= MAX_STEPS_PER_EPISODE:
        waiting = True
        stats = get_completion_stats(graph)
        episodes_data.append({'episode': episode, 'steps': step, 'G': G, 'completed': False})
        logger.log(f"Progress: {stats['visited']}/{stats['total_free']} cells ({stats['percentage']:.1f}%)")
        logger.log_episode_end(episode, step, G, "max steps reached")
        print(f"[DEBUG] episode end, len(traj)={len(trajectory)}")
        _, G_ep = reinforce_update(trajectory, alpha=0.005, gamma=1.0)
        print(f"[DEBUG] reinforce called, G_ep={G_ep}")
        append_return_log(
            episode=episode,
            steps=step,
            G=G,
            completed=False
        )
    
    return graph, state, G, episode, step, waiting


def main():
    """Main game loop"""
    logger.log("=== Grid World RL Started ===")
    mode = "WITH visualization" if ENABLE_VISUALIZATION else "HEADLESS mode"
    logger.log(f"Mode: {mode}")
    logger.log("Controls: SPACE=step, A=auto, R=reset, ESC=quit\n")
    
    # Initialize pygame if visualization enabled
    if ENABLE_VISUALIZATION:
        screen, font, small_font, tiny_font, clock = init_pygame()
    
    # Initialize game state
    scene_graph, s, G, episode, step, waiting = reset_game()
    logger.log_episode_start(episode, START_POSITION)
    
    auto = False
    running = True
    episodes_data = []
    trajectory = []

    while running:
        if ENABLE_VISUALIZATION:
            clock.tick(FPS)
            
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                
                if event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_q, pygame.K_ESCAPE):
                        running = False
                    
                    elif event.key == pygame.K_SPACE:
                        scene_graph, s, G, episode, step, waiting = do_step(
                            scene_graph, s, G, episode, step, waiting, episodes_data, trajectory
                        )      

                    
                    if event.key == pygame.K_a:
                        auto = not auto
                        logger.log(f"Auto mode: {'ON' if auto else 'OFF'}")
                    
                    if event.key == pygame.K_r:
                        logger.log("\n>>> MANUAL RESET <<<")
                        if step > 0:
                            completed = check_completion(scene_graph)
                            episodes_data.append({'episode': episode, 'steps': step, 'G': G, 'completed': completed})
                            logger.log_episode_end(episode, step, G, "manual reset")
                        scene_graph, s, G, episode, step, waiting = reset_game()
                        logger.log_episode_start(episode, START_POSITION)
                        auto = False
            
            # Auto mode
            if auto and pygame.time.get_ticks() % AUTO_STEP_DELAY < 20:
                scene_graph, s, G, episode, step, waiting = do_step(
                    scene_graph, s, G, episode, step, waiting, episodes_data, trajectory
                )
            
            # Draw everything
            draw_grid(screen, font, small_font, tiny_font, scene_graph, s, G, episode, waiting, step)
        
        else:
            # Headless mode - run episodes automatically
            scene_graph, s, G, episode, step, waiting = do_step(
                scene_graph, s, G, episode, step, waiting, episodes_data, trajectory
            )
            
            # Limit number of episodes in headless mode
            if episode > HEADLESS_EPISODES:
                running = False
            
            # Small delay to not overload CPU
            time.sleep(0.001)
    
    # Log final statistics
    logger.log_statistics(episodes_data)
    logger.log("\n=== Game Ended ===")
    
    if ENABLE_VISUALIZATION:
        pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
