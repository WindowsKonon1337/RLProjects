
import matplotlib
import matplotlib.pyplot as plt
import json
import os
from typing import List, Dict


matplotlib.use('Agg')

def plot_training_results(history: List[Dict], filename: str):
    """
    Plots training results from a history list.
    history: List of dicts like {'episode': int, 'reward': float, 'visited_ratio': float}
    filename: Path to save the plot (e.g., 'training_plot.png')
    """
    if not history:
        print("No history to plot.")
        return

    episodes = [x['episode'] for x in history]
    rewards = [x['reward'] for x in history]
    visited_ratios = [x.get('visited_ratio', 0) for x in history]

    plt.figure(figsize=(12, 6))

    # Plot 1: Total Reward
    plt.subplot(1, 2, 1)
    plt.plot(episodes, rewards, label='Total Reward', color='blue', alpha=0.7)

    # Calculate moving average (window 20)
    if len(rewards) >= 20:
        ma = []
        window = 20
        for i in range(len(rewards)):
            start = max(0, i - window + 1)
            ma.append(sum(rewards[start:i+1]) / (i - start + 1))
        plt.plot(episodes, ma, label='Moving Avg (20)', color='orange', linewidth=2)

    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Training Reward Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 2: Visited Ratio
    plt.subplot(1, 2, 2)
    plt.plot(episodes, visited_ratios, label='Visited Ratio', color='green', alpha=0.7)
    plt.xlabel('Episode')
    plt.ylabel('Visited Ratio (0-1)')
    plt.title('Exploration Efficiency')
    plt.ylim(0, 1.05)
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    try:
        plt.savefig(filename)
        print(f"Training plot saved to {filename}")
    except Exception as e:
        print(f"Failed to save plot: {e}")
    finally:
        plt.close()

def save_stats_json(history: List[Dict], filename: str):
    """Saves history list to JSON file."""
    try:
        with open(filename, 'w') as f:
            json.dump(history, f, indent=4)
        print(f"Training stats saved to {filename}")
    except Exception as e:
        print(f"Failed to save stats JSON: {e}")

def load_stats_json(filename: str) -> List[Dict]:
    """Loads history list from JSON file."""
    if not os.path.exists(filename):
        return []
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Failed to load stats JSON: {e}")
        return []