from typing import cast
import numpy as np
from gymnasium import ObservationWrapper, spaces
from minigrid.minigrid_env import MiniGridEnv
from minigrid.core.grid import Grid
from minigrid.core.world_object import Goal
from minigrid.core.mission import MissionSpace
from minigrid.envs.doorkey import DoorKeyEnv
from config import *
from envs.wrappers import ConvWrapper


def mission():
    return "Reach the goal"

class SimpleEnv(DoorKeyEnv):
    def __init__(self, render_mode="human"):
        super().__init__(
            max_steps=CURRICULUM_STEPS["key"],
            see_through_walls=False,
            render_mode=render_mode
        )

    def step(self, action):
        obs, _, terminated, truncated, info = super().step(action)

        if terminated:
            reward = CURRICULUM_REWARDS["goal"]
        else:
            reward = CURRICULUM_REWARDS["normal"]

        return obs, reward, terminated, truncated, info


def make_key_env():
    env = SimpleEnv(render_mode="rgb_array")
    env = ConvWrapper(env)
    return env