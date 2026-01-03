from minigrid.envs.fourrooms import FourRoomsEnv
from config import *
from envs.wrappers import ConvWrapper


def mission():
    return "Reach the goal"

class SimpleEnv(FourRoomsEnv):
    def __init__(self, render_mode="human"):
        super().__init__(
            max_steps=CURRICULUM_STEPS["four_rooms"],
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


def make_four_rooms_env():
    env = SimpleEnv(render_mode="rgb_array")
    env = ConvWrapper(env)
    return env