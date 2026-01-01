from minigrid.envs.crossing import CrossingEnv

from envs.wrappers import ConvWrapper
from config import CURRICULUM_STEPS

class CrossingModified(CrossingEnv):
    def __init__(self, render_mode="human"):
        super().__init__(
            render_mode="rgb_array",
            size=10,
            num_crossings=3,
            max_steps=CURRICULUM_STEPS["crossing"],

        )

    def step(self, action):
        obs, _, terminated, truncated, info = super().step(action)

        cell = self.grid.get(*self.agent_pos)
        if cell is not None and cell.type == "lava":
            reward = -5
        elif terminated:
            reward = 1

        else:
            reward = -0.01



        return obs, reward, terminated, truncated, info

def make_lava_crossing_env():
    env = CrossingModified()
    env = ConvWrapper(env)
    return env