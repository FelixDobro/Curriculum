import random
from typing import Any

from gymnasium.core import ObsType
from minigrid.envs.crossing import CrossingEnv

from envs.wrappers import ConvWrapper
from config import CURRICULUM_STEPS, CURRICULUM_REWARDS


class CrossingModified(CrossingEnv):
    '''def __init__(self, render_mode="rgb_array", min_crossings=1, max_crossings=3, **kwargs):
        self.min_crossings = min_crossings
        self.max_crossings = max_crossings

        super().__init__(num_crossings=max_crossings, render_mode=render_mode, **kwargs)

    def _gen_grid(self, width, height):

        # Nur so funktioniert seed=42 beim reset()
        self.num_crossings = self.np_random.integers(
            self.min_crossings,
            self.max_crossings + 1
        )

        # 2. Jetzt die Grid-Generierung der Original-Klasse aufrufen.
        # Da wir self.num_crossings gerade geändert haben, nutzt
        # die Original-Methode jetzt den neuen Wert.
        super()._gen_grid(width, height)'''

    def __init__(self, render_mode="rgb_array"):
        # n_obstacles steuert, wie "zugemüllt" der Raum ist
        super().__init__(render_mode=render_mode)

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

def make_lava_crossing_env(render_mode="rgb_array"):
    env = CrossingModified(render_mode)
    env = ConvWrapper(env)
    return env