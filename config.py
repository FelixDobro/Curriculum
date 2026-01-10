# paths.py (liegt direkt im Project Root)
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOG_DIR = PROJECT_ROOT / "logs/big_curriculum"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints/big_curriculum"
Path.mkdir(CHECKPOINTS_DIR, exist_ok=True)


## SETUP

'''All of the envs in this list will be used for Training
Options are:

["simple", "multiroom", "big_multiroom", "cluster", "lava_maze",
    "crossing", "dungeon", "key", "locked_room", "four_rooms", "color"]
    
NOTE: This list is both used for every script not only training
    '''


CURRICULUM = ["simple", "multiroom", "cluster", "big_multiroom",
    "crossing", "key", "locked_room", "four_rooms", "color", "lava_maze"]


## Define the number of steps for each env after which truncation is reaches (only relevant for
## elements defined above in the CURRICULUM list)
CURRICULUM_STEPS = {
    "simple": 20,
    "multiroom": 100,
    "lava_maze": 125,
    "big_multiroom": 500,
    "cluster": 100,
    "crossing": 75,
    "dungeon": 1500,
    "key": 75,
    "locked_room": 200,
    "four_rooms": 175,
    "color": 125
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

MODEL_VERSION = 374
MODEL_DIR = CHECKPOINTS_DIR / f"model{MODEL_VERSION}.pt"


## Video settings

FPS = 10
NUM_VIDEOS = 4


## ADVANCED SETUP (You don't need to think about that)

CHUNK_SIZE = 200
NUM_ACTIONS = 6
LEARNING_RATE = 1e-4
GAMMA = 0.99
HIDDEN_DIMS = 256
## Dimensions after conv
EMBEDDING_DIM = 264


