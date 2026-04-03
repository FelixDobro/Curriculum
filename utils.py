from typing import List

import torch

from envs.four_rooms import make_four_rooms_env
from envs.key_env import make_key_env, make_simple_key
from envs.color_env import make_color_env
from envs.lava_maze import *
from envs.dungeons import * 
from envs.multiroom import make_room_env
from envs.locked_room import make_locked_room    
from envs.random_goal_agent import make_env
from envs.big_multi_room import make_big_multi_env

from envs.maze import *
import time


env_functions = {
    "empty": make_env,
    "multiroom": make_room_env,
    "big_multiroom": make_big_multi_env,
    "key": make_key_env,
    "simple_key": make_simple_key,
    "locked_room": make_locked_room,
    "color": make_color_env,
    "four_rooms": make_four_rooms_env,
    "lava_0": lava_0,
    "lava_1": lava_1,
    "lava_2": lava_2,
    "lava_3": lava_3,
    "lava_4": lava_4,
    "key_dungeon": make_key_dungeon,
    "easy_memory": make_easy_memory,
    "memory": make_memory,
    "advanced_memory": make_advanced_memory,
    "obstacles": make_obstacles,
    "lava_goal": make_lava_goal,
    "dungeon_1": make_dungeon_1,
    "dungeon_2": make_dungeon_2,
    "dungeon_3": make_dungeon_3,
    "maze_0": make_maze_diff0,
    "maze_1": make_maze_diff1,
    "maze_2": make_maze_diff2,
    "maze_3": make_maze_diff3,
    "maze_4": make_maze_diff4,
    "maze_5": make_maze_diff5
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