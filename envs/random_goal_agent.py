from minigrid.minigrid_env import MiniGridEnv
from minigrid.core.grid import Grid
from minigrid.core.world_object import Goal
from minigrid.core.mission import MissionSpace
from minigrid.wrappers import FlatObsWrapper, FullyObsWrapper

from config import CURRICULUM_STEPS, CURRICULUM_REWARDS
from envs.wrappers import ConvWrapper


def mission():
    return "Reach the goal"

class SimpleEnv(MiniGridEnv):
    def __init__(self, size=40, render_mode="rgb_array"):
        mission_space = MissionSpace(mission)
        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            max_steps=CURRICULUM_STEPS["simple"],
            see_through_walls=False,
            render_mode=render_mode
        )

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)

        # Platziere das Goal fix (z. B. unten rechts – aber innerhalb der Wände)
        self.goal_pos =self.place_obj(Goal())
        self.agent_pos = self.place_agent()

    def step(self, action):
        obs, _, terminated, truncated, info = super().step(action)

        if terminated:
            reward = CURRICULUM_REWARDS["goal"]
        else:
            reward = CURRICULUM_REWARDS["normal"]

        return obs, reward, terminated, truncated, info



def make_env(size=8):
    env = SimpleEnv(render_mode="rgb_array", size=size)
    env = ConvWrapper(env)
    return env