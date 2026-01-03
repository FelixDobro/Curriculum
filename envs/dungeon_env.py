from typing import cast
import numpy as np
from gymnasium import spaces
from minigrid.core.constants import COLOR_NAMES
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.minigrid_env import MiniGridEnv
from minigrid.core.world_object import Wall, Door, Key, Goal, Lava
from envs.wrappers import ConvWrapper
from config import CURRICULUM_STEPS, CURRICULUM_REWARDS


class BossLevelEnv(MiniGridEnv):
    """
    Der Endgegner. Kombiniert Cluster, Crossing, Key/Door und Multiroom.
    Aufbau: 4 Quadranten, die nacheinander durchlaufen werden müssen.
    """

    def __init__(self, size=24, render_mode="rgb_array", **kwargs):
        # Wir brauchen genug Platz für 4 Räume, also min 20x20
        self.size = size
        mission_space = MissionSpace(mission_func=lambda: "survive the dungeon")

        # Wir nutzen den Dungeon-Step-Count, weil das hier lange dauern kann
        super().__init__(
            mission_space=mission_space,
            width=size,
            height=size,
            max_steps=CURRICULUM_STEPS.get("dungeon", 400),  # Fallback falls key fehlt
            render_mode=render_mode,
            see_through_walls=False,
            **kwargs,
        )

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)

        # --- Layout: 4 Quadranten (2x2) ---
        # Wir ziehen ein Kreuz aus Wänden in die Mitte
        mid_x = width // 2
        mid_y = height // 2

        self.grid.vert_wall(mid_x, 0, height)
        self.grid.horz_wall(0, mid_y, width)

        # --- Definition der Räume (Quadranten) ---
        # Room 1: Top-Left  (Start)
        # Room 2: Top-Right (Crossing)
        # Room 3: Bottom-Right (Lava/Key)
        # Room 4: Bottom-Left (Goal)

        # --- ROOM 1: CLUSTER (Top-Left) ---
        # Bereich: x[1, mid_x-1], y[1, mid_y-1]
        # Wir füllen diesen Bereich mit zufälligen Wänden
        n_cluster_walls = 15
        for _ in range(n_cluster_walls):
            rx = self._rand_int(1, mid_x - 1)
            ry = self._rand_int(1, mid_y - 1)
            # Nicht direkt neben Türen blockieren (simple Heuristik: Mitte freihalten ist schwer,
            # wir verlassen uns drauf, dass place_obj checkt ob frei ist, aber wir setzen einfach Wände)
            # Um sicherzugehen, dass es lösbar ist, löschen wir später Wände um Agent/Tür,
            # oder wir nutzen place_obj rekursiv. Hier quick & dirty:
            self.grid.set(rx, ry, Wall())

        # Agent Start Position (irgendwo in Room 1)
        self.place_agent(size=(mid_x, mid_y))  # Sucht freien Platz in Top-Left

        # Schlüssel 1 (Yellow) für Tür 1
        key1 = Key("yellow")
        self.place_obj(key1, top=(0, 0), size=(mid_x, mid_y))

        # Tür 1 (Yellow) verbindung Room 1 -> Room 2
        # Position: In der vertikalen Wand bei mid_x, irgendwo oben
        door1_y = self._rand_int(1, mid_y - 1)
        self.grid.set(mid_x, door1_y, Door("yellow", is_locked=True))
        # Sicherstellen, dass vor der Tür keine Wall ist
        self.grid.set(mid_x - 1, door1_y, None)
        self.grid.set(mid_x + 1, door1_y, None)

        # --- ROOM 2: CROSSING (Top-Right) ---
        # Bereich: x[mid_x+1, width-1], y[1, mid_y-1]
        # Wir bauen vertikale Hindernis-Linien
        for i in range(mid_x + 2, width - 2, 2):
            # Lücke zufällig
            gap = self._rand_int(1, mid_y - 1)
            self.grid.vert_wall(i, 1, mid_y - 1)
            self.grid.set(i, gap, None)  # Lücke machen

        # Tür 2 (Red) Verbindung Room 2 -> Room 3
        # Position: In der horizontalen Wand bei mid_y, rechts
        door2_x = self._rand_int(mid_x + 1, width - 1)
        self.grid.set(door2_x, mid_y, Door("red", is_locked=True))
        self.grid.set(door2_x, mid_y - 1, None)
        self.grid.set(door2_x, mid_y + 1, None)

        # --- ROOM 3: LAVA & KEY (Bottom-Right) ---
        # Bereich: x[mid_x+1, width-1], y[mid_y+1, height-1]
        # Hier muss der Schlüssel für Tür 2 (Red) liegen!
        # Moment, wir brauchen den roten Schlüssel, um HIER REIN zu kommen?
        # Nein, Logikfehler im Plan.
        # Plan: Wir sind in Room 2, wir wollen nach Room 3. Also muss Key 2 in Room 2 liegen.

        # Korrektur: Platzieren wir Key 2 (Red) in Room 2 (Crossing)
        key2 = Key("red")
        self.place_obj(key2, top=(mid_x + 1, 0), size=(mid_x - 1, mid_y))

        # Room 3 ist der "Vorhof" zum Ziel. Machen wir hier Lava Challenges.
        # Wir brauchen eine Tür zu Room 4 (Green).
        door3_x = self._rand_int(1, mid_x - 1)
        # Tür ist in der vertikalen Wand? Nein, Room 3 ist Bottom-Right, Room 4 ist Bottom-Left.
        # Verbindung ist bei mid_x (vertikale Wand unten).
        door3_y = self._rand_int(mid_y + 1, height - 1)
        self.grid.set(mid_x, door3_y, Door("green", is_locked=True))
        self.grid.set(mid_x - 1, door3_y, None)
        self.grid.set(mid_x + 1, door3_y, None)

        # Key 3 (Green) muss in Room 3 sein
        key3 = Key("green")
        self.place_obj(key3, top=(mid_x + 1, mid_y + 1), size=(mid_x - 1, mid_y - 1))

        # Ein paar Lava Pools in Room 3 zur Abschreckung
        for _ in range(6):
            self.place_obj(Lava(), top=(mid_x + 1, mid_y + 1), size=(mid_x - 1, mid_y - 1))

        # --- ROOM 4: GOAL (Bottom-Left) ---
        # Bereich: x[1, mid_x-1], y[mid_y+1, height-1]
        # Einfach nur das Ziel.
        self.place_obj(Goal(), top=(1, mid_y + 1), size=(mid_x - 1, mid_y - 1))

        self.mission = "navigate the dungeon"

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)

        # Custom Reward Logic aus deiner Config
        cell = self.grid.get(*self.agent_pos)

        if terminated and reward > 0:  # Goal erreicht
            reward = CURRICULUM_REWARDS["goal"]
        elif cell is not None and cell.type == "lava":
            reward = CURRICULUM_REWARDS["lava"]
            terminated = True  # Optional: Instant death bei Lava?
        else:
            reward = CURRICULUM_REWARDS["normal"]

        return obs, reward, terminated, truncated, info


def make_boss_env():
    env = BossLevelEnv(size=24, render_mode="rgb_array")
    env = ConvWrapper(env)
    return env