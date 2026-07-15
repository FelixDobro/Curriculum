
from pathlib import Path
import torch

PROJECT_ROOT = Path(__file__).resolve().parent
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(DEVICE)
LOG_DIR = PROJECT_ROOT / "logs/big_c"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints/big_c"
Path.mkdir(CHECKPOINTS_DIR, exist_ok=True, parents=True)


## SETUP

'''All of the envs in this list will be used for Training
Options are:

["empty", "multiroom", "big_multiroom", "cluster", "lava_maze",
    ,"maze_1", "maze_2", "maze_3"]
    
NOTE: This list is both used for every script not only training
    '''

CURRICULUM = ["empty", "multiroom", "key", "locked_room", "easy_memory", "lava_0", "lava_1", "lava_2", "lava_maze_0",
 "lava_maze_1", "maze_1", "lava_goal", "lava_maze_key","lava_maze_key_locked", "obstacles_1", "lava_key_obstacles", "memory_obstacles"]


## Define the number of steps for each env after which truncation is reaches (only relevant for
## elements defined above in the CURRICULUM list)
CURRICULUM_STEPS = {
    "empty": 50,
    "multiroom": 250,
    "big_multiroom": 2000,
    "key": 250,
    "simple_key": 50,
    "locked_room": 600,
    "four_rooms": 175,
    "easy_memory": 200,
    "memory": 500,
    "medium_memory": 1000,
    "advanced_memory": 1500,
    "key_dungeon": 500,
    "obstacle_key_simple": 300,
    "obstacle_key": 500,
    "obstacles": 300,
    "obstacles_1": 500,
    "obstacles_2":750,
    "combined": 1500,
    "lava_goal": 400,
    "lava_key_obstacles": 500,
    "memory_obstacles": 600,
    "lava_maze_key": 1000,
    "lava_maze_key_locked": 1200,
    "dungeon_1": 1000,
    "dungeon_2": 2000,
    "dungeon_3": 2000,
    "color": 300,
    "lava_0":100,
    "lava_1":150,
    "lava_2":250,
    "lava_maze_0": 200,
    "lava_maze_1": 300,
    "maze_0": 150,
    "maze_1": 250,
    "maze_2": 360,
    "maze_3": 450,
    "maze_4": 600,
    "maze_5":1000
}


CURRICULUM_REWARDS = {
    "normal": -0.001,
    "lava": -0.05,
    "goal": 1
}

PPO_EPOCHS = 5
NUM_ENVS = 10
START_SEED = 3
NUM_ROUNDS = 5
NUM_SAMPLES_PER_ROUND = 50_000_000
SAVE_EVERY_SAMPLES = 10_000_000

## Model checkpoint that will be used for inference scripts

MODEL_VERSION = "_last"
RUN =  1
MODEL_DIR = CHECKPOINTS_DIR / f"_run_{RUN}" / f"model{MODEL_VERSION}.pt"


## Eval settings

FPS = 5
NUM_VIDEOS = 1
TEMPERATURE = 1
NUM_EPISODES_EVAL = 50
 

## ADVANCED SETUP 
TEACHER_EPSILON = 0.15
CHUNK_SIZE = 400
NUM_ACTIONS = 6
LEARNING_RATE = 1e-4
MIN_LR = 1e-4
GAMMA = 0.99
MIN_BETA = 1e-4
GAE_LAMBDA = 0.95
HIDDEN_DIMS = 1024
## Dimensions after conv
EMBEDDING_DIM = 512




SLIDING_STEPS = 500  #Number of steps before env resests
PERFORMANCE_SAMPLE_SIZE = 2000 # Number of EPISODES before checking wheter to grade down/up
WINDOW_SIZE = 1 #irrelevant
MAX_SIZE=30  ##
MIN_SIZE = 30 ## Refer both to size of env
UPDTAE_UP = 0.95 # Update Threshold
UPDATE_DOWN = 0.5 # Update Threshold