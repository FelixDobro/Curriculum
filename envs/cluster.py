import numpy as np
from gymnasium import ObservationWrapper, spaces
from minigrid.core.world_object import Goal, Wall
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.minigrid_env import MiniGridEnv

from envs.wrappers import ConvWrapper

from config import CURRICULUM_STEPS
class ClutteredGoalEnv(MiniGridEnv):
    """
    Ein Raum voller zufälliger Wände/Hindernisse.
    Der Agent muss das Ziel finden, ohne steckenzubleiben.
    """
    def __init__(self, size=25, n_obstacles=30, **kwargs):
        self.n_obstacles = n_obstacles
        mission_space = MissionSpace(mission_func=lambda: "navigate clutter")
        super().__init__(
            mission_space=mission_space,
            width=size,
            height=size,
            max_steps=CURRICULUM_STEPS["cluster"],
            **kwargs,
        )

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)

        # Agent und Ziel zufällig platzieren
        self.place_agent()
        self.place_obj(Goal())

        # Zufällige Hindernisse (Wände) platzieren
        for i in range(self.n_obstacles):
            # Wir versuchen, Wall-Objekte zu platzieren
            # place_obj sucht automatisch nach einem freien Platz
            try:
                self.place_obj(Wall())
            except:
                pass # Falls kein Platz mehr ist, egal

        self.mission = "navigate clutter"

# Wrapper
class SimpleClutterEnv(ClutteredGoalEnv):
    def __init__(self, render_mode="human"):
        # n_obstacles steuert, wie "zugemüllt" der Raum ist
        super().__init__(size=14, n_obstacles=35, render_mode=render_mode)

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        if terminated: reward = 1
        else: reward = -0.01
        return obs, reward, terminated, truncated, info



def make_cluster_env():
    env = SimpleClutterEnv(render_mode="rgb_array")
    env = ConvWrapper(env)
    return env