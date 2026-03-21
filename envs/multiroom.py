from typing import cast
import numpy as np
from gymnasium import ObservationWrapper, spaces
from minigrid.envs.multiroom import MultiRoomEnv
from config import *
from envs.wrappers import ConvWrapper


def mission():
    return "Reach the goal"

class SimpleEnv(MultiRoomEnv):
    def __init__(self, render_mode="human"):
        super().__init__(
            max_steps=CURRICULUM_STEPS["multiroom"],
            see_through_walls=False,
            render_mode=render_mode,
            minNumRooms=2,
            maxNumRooms=4,
        )

    def step(self, action):
        obs, _, terminated, truncated, info = super().step(action)

        if terminated:
            reward = CURRICULUM_REWARDS["goal"]
            info["success"] = True
        else:
            info["success"] = False
            reward = CURRICULUM_REWARDS["normal"]
            
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        info["success"] = False 

        return obs, info

def make_room_env():
    env = SimpleEnv(render_mode="rgb_array")
    env = ConvWrapper(env)
    return env