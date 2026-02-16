"""
Configuration file for Grid World RL Game
All game parameters, colors, and settings are defined here.
"""

# === GAME PARAMETERS ===
GRID_SIZE = 5 # Size of the game grid (10x10)
OBSTACLE_PROB = 0.1 # Probability of obstacle generation
START_POSITION = (0, GRID_SIZE - 1)  # Agent always starts at top-left corner

# === VISUAL PARAMETERS ===
CELL_SIZE = 50  # Size of each cell in pixels
GAP = 2  # Gap between cells
MARGIN = 50  # Margin from left/right
TOP_MARGIN = 120  # Top margin for info display
LEGEND_WIDTH = 280  # Width of legend sidebar
INFO_HEIGHT = 100  # Height of bottom info area

# === REWARD PARAMETERS ===
REWARD_VISITED = -0.5  # Penalty for revisiting a cell
REWARD_NEW = 1.0  # Reward for visiting a new cell
REWARD_COMPLETION = 0  # Bonus reward for visiting all non-obstacle cells

# === COLORS ===
# Background
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
ENABLE_VISUALIZATION = False  # Set to False to run without pygame visualization
ENABLE_LOGGING = True  # Set to False to disable logging to file
LOG_FILE = "game_log.log"  # Log file name
VERBOSE_CONSOLE = True  # Print detailed info to console

# === EPISODE SETTINGS ===
MAX_STEPS_PER_EPISODE = 500  # Maximum steps before episode ends
HEADLESS_EPISODES = 1000 # Number of episodes to run in headless mode 
