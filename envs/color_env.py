from __future__ import annotations
from gymnasium import spaces
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Goal, Key, Door, Wall
from minigrid.minigrid_env import MiniGridEnv

from config import CURRICULUM_REWARDS
from envs.wrappers import ConvWrapper
import random


class ColorEnv(MiniGridEnv):
    def __init__(
            self,
            size=8,
            num_doors=3,
            render_mode="rgb_array",
            max_steps=100,
            **kwargs,
    ):
        self.num_doors = num_doors
        self.size = size
        mission_space = MissionSpace(mission_func=lambda: "open the correct door")

        super().__init__(
            mission_space=mission_space,
            width=size,
            height=size,
            max_steps=max_steps,
            render_mode=render_mode,
            see_through_walls=False,
            **kwargs,
        )

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)

        colors = ["red", "green", "blue", "purple", "yellow", "grey"]

        door_colors = random.sample(colors, self.num_doors)

        target_color = random.choice(door_colors)

        split_idx = height - 4

        for i in range(1, width - 1):
            self.grid.set(i, split_idx, Wall())

        door_indices = random.sample(range(1, width - 1), self.num_doors)

        for idx, color in zip(door_indices, door_colors):
            self.grid.set(idx, split_idx, Door(color, is_locked=True))

            if color == target_color:
                self.grid.set(idx, split_idx + 2, Goal())
            else:
                pass

        self.place_agent(size=(width, split_idx))

        key_obj = Key(target_color)
        self.place_obj(key_obj, size=(width, split_idx))

        self.mission = f"pick up the {target_color} key and open the {target_color} door"

    def step(self, action):
        obs, _, terminated, truncated, info = super().step(action)

        if terminated:
            reward = CURRICULUM_REWARDS["goal"]
        else:
            reward = CURRICULUM_REWARDS["normal"]

        return obs, reward, terminated, truncated, info

def make_color_env():
    env = ColorEnv(size=8, num_doors=3, render_mode="rgb_array")
    env = ConvWrapper(env)
    return env