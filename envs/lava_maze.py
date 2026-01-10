import random
from typing import Any

from gymnasium.core import ObsType
from minigrid.envs.crossing import CrossingEnv

from envs.wrappers import ConvWrapper
from config import CURRICULUM_STEPS, CURRICULUM_REWARDS


class LavaMaze(CrossingEnv):
    def __init__(self, render_mode="rgb_array", **kwargs):
        self.min_crossings = 3
        self.max_crossings = 8
        size = 17
        self.max_steps = CURRICULUM_STEPS["lava_maze"]
        super().__init__(num_crossings=self.min_crossings, size=size,render_mode=render_mode, **kwargs)

    def _gen_grid(self, width, height):

        # Nur so funktioniert seed=42 beim reset()
        self.num_crossings = self.np_random.integers(
            self.min_crossings,
            self.max_crossings + 1
        )
#
        super()._gen_grid(width, height)



    def step(self, action):
        obs, _, terminated, truncated, info = super().step(action)

        cell = self.grid.get(*self.agent_pos)
        if cell is not None and cell.type == "lava":
            reward = CURRICULUM_REWARDS["lava"]
        elif terminated:
            reward = CURRICULUM_REWARDS["goal"]
        else:
            reward = CURRICULUM_REWARDS["normal"]



        return obs, reward, terminated, truncated, info

def make_lava_maze_env(render_mode="rgb_array"):
    env = LavaMaze(render_mode)
    env = ConvWrapper(env)
    return env