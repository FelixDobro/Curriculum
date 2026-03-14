from typing import List

import torch

from envs.four_rooms import make_four_rooms_env
from envs.key_env import make_key_env, make_simple_key
from envs.dungeon_env import make_boss_env
from envs.crossing import make_lava_crossing_env
from envs.color_env import make_color_env
from envs.lava_maze import make_lava_maze_env
from envs.multiroom import make_room_env
from envs.locked_room import make_locked_room    
from envs.random_goal_agent import make_env
from envs.big_multi_room import make_big_multi_env
from envs.hard_env import *
from envs.maze import *
import time


env_functions = {
    "empty": make_env,
    "multiroom": make_room_env,
    "big_multiroom": make_big_multi_env,
    "crossing": make_lava_crossing_env,
    "dungeon": make_boss_env,
    "key": make_key_env,
    "simple_key": make_simple_key,
    "locked_room": make_locked_room,
    "color": make_color_env,
    "four_rooms": make_four_rooms_env,
    "lava_maze": make_lava_maze_env,
    "maze_0": make_maze_diff0,
    "maze_1": make_maze_diff1,
    "maze_2": make_maze_diff2,
    "maze_3": make_maze_diff3,
    "maze_4": make_maze_diff4,
    "maze_5": make_maze_diff5,
    "difficulty1": difficulty_1,
    "difficulty2": difficulty_2,
    "difficulty3": difficulty_3,
    "difficulty4": difficulty_4,
    "difficulty5": difficulty_5,
    "difficulty6": difficulty_6,
    "difficulty7": difficulty_7,
}

def map_envs(names):
    curriculum_names = {}

    curriculum_functions = {}

    for i,name in enumerate(names):
        if name not in env_functions:
            raise ValueError(f"{name} not a known env name, maybe wrong spelling? Options are: \n {list(env_functions.keys())}")
        curriculum_names[i] = name

        curriculum_functions[i] = env_functions[name]
    return curriculum_names, curriculum_functions



def log_time(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        ende = time.time()
        print(f"{func}: {ende - start} Seconds")
        return result
    return wrapper

    
def forward_pass(state_chunk, learning_model, dones_chunk, learning_hidden):
    policy_logits_list: List[torch.Tensor] = []
    extrinsic_val_list: List[torch.Tensor] = []

    for t_step in range(state_chunk.size(1)):
        step_state = state_chunk[:, t_step]
    
        v_ext, p, learning_hidden = learning_model(step_state, learning_hidden)
        
        policy_logits_list.append(p.squeeze(1))
        extrinsic_val_list.append(v_ext.squeeze(1).squeeze(-1))
        
        done_mask = 1.0 - dones_chunk[:, t_step].float().unsqueeze(-1)
        learning_hidden = learning_hidden * done_mask
    return policy_logits_list, extrinsic_val_list