# paths.py (liegt direkt im Project Root)
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(DEVICE)
LOG_DIR = PROJECT_ROOT / "logs/multiroom_b"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints/multiroom_b"
Path.mkdir(CHECKPOINTS_DIR, exist_ok=True, parents=True)


## SETUP

'''All of the envs in this list will be used for Training
Options are:

["empty", "multiroom", "big_multiroom", "cluster", "lava_maze",
    ,"maze_1", "maze_2", "maze_3"]
    
NOTE: This list is both used for every script not only training
    '''

CURRICULUM = ["big_multiroom"]


## Define the number of steps for each env after which truncation is reaches (only relevant for
## elements defined above in the CURRICULUM list)
CURRICULUM_STEPS = {
    "empty": 50,
    "multiroom": 250,
    "big_multiroom": 1000,
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
    "lava_goal": 500,
    "dungeon_1": 1000,
    "dungeon_2": 2000,
    "dungeon_3": 2000,
    "color": 300,
    "lava_0":50,
    "lava_1":100,
    "lava_2":125,
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

SAVE_EVERY = 64
UPDATE_PRINT = 20
NUM_ENVS = 12
NUM_ROUNDS = 5
NUM_SAMPLES_PER_ROUND = 5000000

## Model checkpoint that will be used for inference scripts

MODEL_VERSION = 75
MODEL_DIR = CHECKPOINTS_DIR / f"model{MODEL_VERSION}.pt"


## Eval settings

FPS = 5
NUM_VIDEOS = 5
TEMPERATURE = 1
NUM_EPISODES_EVAL = 100


## ADVANCED SETUP 

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