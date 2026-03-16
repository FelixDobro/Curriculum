# paths.py (liegt direkt im Project Root)
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(DEVICE)
LOG_DIR = PROJECT_ROOT / "logs/Sliding"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints/Sliding"
Path.mkdir(CHECKPOINTS_DIR, exist_ok=True, parents=True)


## SETUP

'''All of the envs in this list will be used for Training
Options are:

["empty", "multiroom", "big_multiroom", "cluster", "lava_maze",
    "crossing", "dungeon", "key", "locked_room", "four_rooms", "color",
    "difficulty1", "difficulty2", "difficulty3", "difficulty4", "difficulty5", 
    "difficulty6", "difficulty7", "maze_0", "maze_1", "maze_2", "maze_3"]
    
NOTE: This list is both used for every script not only training
    '''


CURRICULUM = ["empty", "color", "simple_key", "key"]

## Define the number of steps for each env after which truncation is reaches (only relevant for
## elements defined above in the CURRICULUM list)
CURRICULUM_STEPS = {
    "empty": 30,
    "multiroom": 150,
    "lava_maze": 125,
    "big_multiroom": 1000,
    "crossing": 75,
    "dungeon": 1500,
    "key": 250,
    "simple_key": 50,
    "locked_room": 600,
    "four_rooms": 175,
    "color": 300,
    "maze_0": 200,
    "maze_1": 300,
    "maze_2": 600,
    "maze_3": 800,
    "maze_4": 1000,
    "maze_5":1000,
    "difficulty1": 150,
    "difficulty2": 400,
    "difficulty3": 350,
    "difficulty4": 400,
    "difficulty5": 400, 
    "difficulty6":450,
    "difficulty7":500
}


CURRICULUM_REWARDS = {
    "normal": -0.001,
    "lava": -0.05,
    "goal": 1
}

PPO_EPOCHS = 5

SAVE_EVERY = 100
UPDATE_PRINT = 20
NUM_ENVS = 12

## Model checkpoint that will be used for inference scripts

MODEL_VERSION = 63
MODEL_DIR = CHECKPOINTS_DIR / f"model{MODEL_VERSION}.pt"


## Eval settings

FPS = 7
NUM_VIDEOS = 10
TEMPERATURE = 1
NUM_EPISODES_EVAL = 20


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




SLIDING_STEPS = 500 
PERFORMANCE_SAMPLE_SIZE = 5000
WINDOW_SIZE = 1
MAX_SIZE=20
MIN_SIZE = 4
UPDTAE_UP = 0.9
UPDATE_DOWN = 0.5