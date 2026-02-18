"""
Configuration file for Grid World RL Game
All game parameters, colors, and settings are defined here.
"""

# === REPRODUCIBILITY ===
RANDOM_SEED = 42  # Fixed seed for reproducibility (set to None to disable)

# === GAME PARAMETERS ===
GRID_SIZE = 5 + 2  # Size of the game grid (10x10)
OBSTACLE_PROB = 0.1 # Probability of obstacle generation
START_POSITION = (1, 1) # Start position for the agent

# === VISUAL PARAMETERS ===
CELL_SIZE = 50  # Size of each cell in pixels
GAP = 2  # Gap between cells
MARGIN = 50  # Margin from left/right
TOP_MARGIN = 120  # Top margin for info display
LEGEND_WIDTH = 280  # Width of legend sidebar
INFO_HEIGHT = 100  # Height of bottom info area

# === RL HYPERPARAMETERS ===
LR = 1e-4  # Learning rate
GAMMA = 0.99  # Discount factor
CNN_INPUT_REGULARIZATION = 0 # Regularization for first layer
LOGIT_SCALE = 0.05 # "Temperature" for Softmax. Lower = Softer (Flatter) distribution. 0.1 was soft, 0.05 is VERY soft.
TRAIN_MODE = True # Set to False for inference only

# === REWARD PARAMETERS ===
REWARD_VISITED = -0.2  # Penalty for revisiting a cell
REWARD_NEW = 1.0  # Reward for visiting a new cell
REWARD_COMPLETION = 0 # 20.0  # Bonus reward for visiting all non-obstacle cells
REWARD_STEP = -0.01 # Penalty for each step

# === COLORS ===
# Background color
BG_COLOR = (20, 22, 35)

# Grid cells (not observed)
FREE_COLOR = (70, 80, 100)  # Free cells
OBSTACLE_COLOR = (150, 80, 200)  # Obstacles (purple)
VISITED_COLOR = (60, 180, 80)  # Visited cells (green)

# Observed cells (darker variants)
OBSERVED_FREE_COLOR = (100, 90, 20)  # Dark yellow
OBSERVED_VISITED_COLOR = (120, 60, 20)  # Dark orange
OBSERVED_OBSTACLE_COLOR = (100, 20, 20)  # Dark red

# Agent
AGENT_COLOR = (255, 20, 20)  # Bright red

# UI elements
GRID_LINE_COLOR = (90, 100, 120)
TEXT_COLOR = (240, 245, 255)
LEGEND_COLOR = (180, 190, 210)
REWARD_COLOR_POSITIVE = (100, 255, 100)
REWARD_COLOR_NEGATIVE = (255, 100, 100)

# === PYGAME SETTINGS ===
FPS = 60  # Frames per second
AUTO_STEP_DELAY = 100  # Milliseconds between auto steps

# === SIMULATION SETTINGS ===
ENABLE_VISUALIZATION = True  # Set to False to run without pygame visualization
ENABLE_LOGGING = True  # Set to False to disable logging to file
LOG_FILE = "Prj1/results/game_log.log"  # Log file path (used by train.py / eval.py)
LOG_LEVEL = "INFO"  # Logging verbosity: DEBUG, INFO, WARNING, ERROR
VERBOSE_CONSOLE = True  # Print detailed info to console

# === EPISODE SETTINGS ===
MAX_STEPS_PER_EPISODE = 200  # Maximum steps before episode ends
NUM_EPISODES = 30000          # Total number of training episodes
EVAL_EPISODES = 20            # Number of episodes to run during evaluation
HEADLESS_EPISODES = 1000     # Number of episodes to run without visualization (fast forward)

# === MODEL SETTINGS ===
MODEL_TYPE = "HEURISTIC_DFS" # "MLP" or "CNN" or "GNN"
CNN_INPUT_REGULARIZATION =  0 #1e-4 # L2 penalty for CNN input layer

# === WEIGHTS MANAGEMENT ===
WEIGHTS_FILE = f"Prj1/results/policy_weights_{MODEL_TYPE}.pth"
LOAD_EXISTING_WEIGHTS = True
SAVE_WEIGHTS = False


# === TRAINING STATS ===
TRAINING_STATS_FILE = f"Prj1/results/training_stats_{MODEL_TYPE}.json"
INFERENCE_STATS_FILE = f"Prj1/results/inference_stats_{MODEL_TYPE}.json"
TRAINING_PLOT_FILE = f"Prj1/results/training_plot_{MODEL_TYPE}.png"
PLOT_TRAINING_CURVE = False
