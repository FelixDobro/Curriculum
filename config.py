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

["simple", "multiroom", "big_multiroom", "cluster",
    "crossing", "dungeon", "key", "locked_room"]
    
NOTE: This list is both used for every script not only training
    '''


CURRICULUM = ["simple", "multiroom", "cluster",
    "crossing", "key", "locked_room"]


## Define the number of steps for each env after which truncation is reaches (only relevant for
## elements defined above in the CURRICULUM list)
CURRICULUM_STEPS = {
    "simple": 20,
    "multiroom": 100,
    "big_multiroom": 350,
    "cluster": 100,
    "crossing": 125,
    "dungeon": 350,
    "key": 75,
    "locked_room": 250,
}
SAVE_EVERY = 200
UPDATE_PRINT = 20
NUM_ENVS = 10

## Model checkpoint that will be used for inference scripts

MODEL_VERSION = 69
MODEL_DIR = CHECKPOINTS_DIR / f"model{MODEL_VERSION}.pt"


## Video settings

FPS = 100
NUM_VIDEOS = 3


## ADVANCED SETUP (You don't need to think about that)

CHUNK_SIZE = 64
NUM_ACTIONS = 6
LEARNING_RATE = 1e-4
GAMMA = 0.99
HIDDEN_DIMS = 256
## Dimensions after conv
EMBEDDING_DIM = 264


