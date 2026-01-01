from minigrid.envs.lockedroom import LockedRoomEnv

from envs.wrappers import ConvWrapper
from config import CURRICULUM_STEPS

class MyLockedRoom(LockedRoomEnv):
    def __init__(self, render_mode="rgb_array"):
        super().__init__(
            max_steps=CURRICULUM_STEPS["locked_room"],
            see_through_walls=False,
            render_mode=render_mode
        )

    def step(self, action):
        obs, _, terminated, truncated, info = super().step(action)

        if terminated:
            reward = 1
        else:
            reward = -0.01

        return obs, reward, terminated, truncated, info


def make_locked_room():
    env = MyLockedRoom()
    env = ConvWrapper(env)
    return env
