from envs.four_rooms import make_four_rooms_env
from envs.key_env import make_key_env
from envs.dungeon_env import make_boss_env
from envs.crossing import make_lava_crossing_env
from envs.color_env import make_color_env
from envs.lava_maze import make_lava_maze_env
from envs.multiroom import make_room_env
from envs.cluster import make_cluster_env
from envs.locked_room import make_locked_room
from envs.random_goal_agent import make_env
from envs.big_multi_room import make_big_multi_env

env_functions = {
    "simple": make_env,
    "multiroom": make_room_env,
    "big_multiroom": make_big_multi_env,
    "cluster": make_cluster_env,
    "crossing": make_lava_crossing_env,
    "dungeon": make_boss_env,
    "key": make_key_env,
    "locked_room": make_locked_room,
    "color": make_color_env,
    "four_rooms": make_four_rooms_env,
    "lava_maze": make_lava_maze_env
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

