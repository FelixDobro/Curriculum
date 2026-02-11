# paths.py (liegt direkt im Project Root)
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOG_DIR = PROJECT_ROOT / "logs/difficult"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints/difficult"
Path.mkdir(CHECKPOINTS_DIR, exist_ok=True, parents=True)


## SETUP

'''All of the envs in this list will be used for Training
Options are:

["simple", "multiroom", "big_multiroom", "cluster", "lava_maze",
    "crossing", "dungeon", "key", "locked_room", "four_rooms", "color",
    "difficulty1", "difficulty2", "difficulty3", "difficulty4", "difficulty5", 
    "difficulty6", "difficulty7"]
    
NOTE: This list is both used for every script not only training
    '''


CURRICULUM = ["simple", "simple_key","key", "difficulty2", "difficulty3", "difficulty7"]

## Define the number of steps for each env after which truncation is reaches (only relevant for
## elements defined above in the CURRICULUM list)
CURRICULUM_STEPS = {
    "simple": 30,
    "multiroom": 150,
    "lava_maze": 125,
    "big_multiroom": 1000,
    "cluster": 100,
    "crossing": 75,
    "dungeon": 1500,
    "key": 100,
    "simple_key": 50,
    "locked_room": 200,
    "four_rooms": 175,
    "color": 300,
    "difficulty1": 150,
    "difficulty2": 150,
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

SAVE_EVERY = 200
UPDATE_PRINT = 20
NUM_ENVS = 12

## Model checkpoint that will be used for inference scripts

MODEL_VERSION = 21
MODEL_DIR = CHECKPOINTS_DIR / f"model{MODEL_VERSION}.pt"


## Eval settings

FPS = 40
NUM_VIDEOS = 1
NUM_EPISODES_EVAL = 300


## ADVANCED SETUP (You don't need to think about that)

CHUNK_SIZE = 200
NUM_ACTIONS = 6
LEARNING_RATE = 5e-4
GAMMA = 0.99
HIDDEN_DIMS = 512
## Dimensions after conv
EMBEDDING_DIM = 264


