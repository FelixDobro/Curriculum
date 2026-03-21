# paths.py (liegt direkt im Project Root)
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(DEVICE)
LOG_DIR = PROJECT_ROOT / "logs/test"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints/test"
Path.mkdir(CHECKPOINTS_DIR, exist_ok=True, parents=True)


## SETUP

'''All of the envs in this list will be used for Training
Options are:

["empty", "multiroom", "big_multiroom", "cluster", "lava_maze",
    ,"maze_1", "maze_2", "maze_3"]
    
NOTE: This list is both used for every script not only training
    '''

CURRICULUM = ["lava_0", "lava_1", "lava_2", "lava_3", "lava_4", 
"dungeon_1", "empty", "multiroom", "maze_0", "maze_1", "maze_2", "maze_3", "maze_4", "locked_room", "key", "color"]

## Define the number of steps for each env after which truncation is reaches (only relevant for
## elements defined above in the CURRICULUM list)
CURRICULUM_STEPS = {
    "empty": 50,
    "multiroom": 250,
    "big_multiroom": 1000,
    "dungeon": 1500,
    "key": 250,
    "simple_key": 50,
    "locked_room": 600,
    "four_rooms": 175,
    "dungeon_1": 1000,
    "color": 300,
    "lava_0":50,
    "lava_1":100,
    "lava_2":125,
    "lava_3": 200,
    "lava_4": 250,
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

SAVE_EVERY = 50
UPDATE_PRINT = 20
NUM_ENVS = 12

## Model checkpoint that will be used for inference scripts

MODEL_VERSION = 5
MODEL_DIR = CHECKPOINTS_DIR / f"model{MODEL_VERSION}.pt"


## Eval settings

FPS = 5
NUM_VIDEOS = 5
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