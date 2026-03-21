import random
from typing import Any

from gymnasium.core import ObsType
from minigrid.envs.crossing import CrossingEnv

from envs.wrappers import ConvWrapper
from config import CURRICULUM_STEPS, CURRICULUM_REWARDS


class LavaMaze(CrossingEnv):
    def __init__(self, render_mode="rgb_array", size=17, min_crossings=3, max_crossings=8, max_steps=CURRICULUM_STEPS["lava_2"], **kwargs):
        self.min_crossings = min_crossings
        self.max_crossings = max_crossings
 
        super().__init__(num_crossings=self.min_crossings, size=size,render_mode=render_mode, max_steps=max_steps, **kwargs)

    def _gen_grid(self, width, height):

       
        self.num_crossings = random.randint(self.min_crossings, self.max_crossings)
        super()._gen_grid(width, height)


    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        info["success"] = False 

        return obs, info


    
    def step(self, action):
        obs, _, terminated, truncated, info = super().step(action)

        cell = self.grid.get(*self.agent_pos)
        if cell is not None and cell.type == "lava":
            reward = CURRICULUM_REWARDS["lava"]
            info["success"] = False
        elif terminated:
            reward = CURRICULUM_REWARDS["goal"]
            info["success"] = True
        else:
            reward = CURRICULUM_REWARDS["normal"]
            info["success"] = False

        return obs, reward, terminated, truncated, info




def lava_0(rener_mode="rgb_array"):
    env = LavaMaze(render_mode=rener_mode, size=5, min_crossings=1, max_crossings=1, max_steps=CURRICULUM_STEPS["lava_0"])
    env = ConvWrapper(env)
    return env

def lava_1(render_mode="rgb_array"):
    env = LavaMaze(render_mode=render_mode, size=7, min_crossings=1, max_crossings=2, max_steps=CURRICULUM_STEPS["lava_1"])
    env = ConvWrapper(env)
    return env

def lava_2(render_mode="rgb_array"):
    env = LavaMaze(render_mode=render_mode, size=9, min_crossings=2, max_crossings=3, max_steps=CURRICULUM_STEPS["lava_2"])
    env = ConvWrapper(env)
    return env

def lava_3(render_mode="rgb_array"):
    env = LavaMaze(render_mode=render_mode, size=11, min_crossings=2, max_crossings=4, max_steps=CURRICULUM_STEPS["lava_2"])
    env = ConvWrapper(env)
    return env

def lava_4(render_mode="rgb_array"):
    env = LavaMaze(render_mode=render_mode, size=13, min_crossings=3, max_crossings=5, max_steps=CURRICULUM_STEPS["lava_2"])
    env = ConvWrapper(env)
    return env