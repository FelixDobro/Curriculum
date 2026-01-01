from typing import cast

import gymnasium as gym
import numpy as np
from gymnasium import spaces, ObservationWrapper
from minigrid.core.constants import COLOR_NAMES
from minigrid.core.grid import Grid
from minigrid.minigrid_env import MiniGridEnv
from minigrid.core.world_object import Wall, Door, Key, Goal, Ball, Lava, Box
from minigrid.core.mission import MissionSpace

from config import CURRICULUM_STEPS
from envs.wrappers import ConvWrapper


def mission():
    return "Reach the goal"

class GodsDungeon(MiniGridEnv):
    def __init__(self, render_mode=None, size=30, max_steps=CURRICULUM_STEPS["dungeon"]):
        super().__init__(mission_space=MissionSpace(mission),
                         width=size, height=size,
                         max_steps=max_steps,
                         see_through_walls=False,
                         render_mode="rgb_array")

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)
        self.agent_pos = (1,1)
        self.agent_dir = 0

        ## Locked Goal
        self.grid.set(2, 3, Wall())
        self.grid.set(2, 4, Wall())
        self.grid.set(2, 5, Wall())
        self.grid.set(1,5, Wall())
        self.grid.set(1,4, Goal())

        ## separation
        for j in range(10):
            self.grid.set(10, j, Wall())
        for i in range(1, 11):
            self.grid.set(i, 10, Wall())
            self.grid.set(i, 24, Wall())
        for j in range(10, width-1):
            self.grid.set(5,j ,Wall())

        for i in range(1,5):
            self.grid.set(i,15, Wall())
            self.grid.set(i, 20, Wall())


        self.grid.set(5,17, Wall(COLOR_NAMES[0]))
        self.grid.set(1,19, Wall(COLOR_NAMES[0]))
        self.grid.set(2,18, Wall(COLOR_NAMES[0]))

        self.grid.set(1,17, Wall(COLOR_NAMES[0]))
        self.grid.set(1,18, Wall(COLOR_NAMES[0]))

        self.grid.set(1, 25, Wall())
        self.grid.set(1, 27, Box(COLOR_NAMES[0]))

        self.grid.set(3,26, Goal())
        self.grid.set(4,15, Door(COLOR_NAMES[3], is_locked=False))
        self.grid.set(3,12, Box(COLOR_NAMES[3]))
        self.grid.set(2,20, Door(COLOR_NAMES[3], is_locked=False))
        self.grid.set(2, 10, Door(COLOR_NAMES[3], is_locked=False))

        self.grid.set(3, 24, Door(COLOR_NAMES[1], is_locked=False))
        self.grid.set(2, 22, Key(COLOR_NAMES[1]))
        self.grid.set(4, 27, Key(COLOR_NAMES[2]))
        self.grid.set(10, 5, Door(COLOR_NAMES[2], is_locked=True))
        self.grid.set(1, 3, Door(COLOR_NAMES[0], is_locked=True))
        self.grid.set(11, 5, Key(COLOR_NAMES[0]))

    def step(self, action):
        obs, _, terminated, truncated, info = super().step(action)
        if terminated:
            reward = 1
        else:
            reward = -0.01
        cell = self.grid.get(*self.agent_pos)
        if cell is not None and cell.type == "lava":
            reward = -10  # Straf-Reward
            terminated = True

        return obs, reward, terminated, truncated, info




def make_god_dungeon_env():
    env = GodsDungeon(render_mode="rgb_array")
    env = ConvWrapper(env)
    return env