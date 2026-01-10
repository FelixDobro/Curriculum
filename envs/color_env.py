from __future__ import annotations
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Goal, Key, Door, Wall

from minigrid.minigrid_env import MiniGridEnv

from config import CURRICULUM_REWARDS, CURRICULUM_STEPS
from envs.wrappers import ConvWrapper
import random


class ColorEnv(MiniGridEnv):
    def __init__(
            self,
            num_doors=3,
            render_mode="rgb_array",
            max_steps=CURRICULUM_STEPS["color"],
            **kwargs,
    ):
        self.num_doors = num_doors
        self.size = 17

        mission_space = MissionSpace(mission_func=lambda: "open the correct door")

        super().__init__(
            mission_space=mission_space,
            width=self.size,
            height=self.size,
            max_steps=max_steps,
            render_mode=render_mode,
            see_through_walls=False,
            **kwargs,
        )

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)

        colors = ["red", "green", "blue", "purple", "yellow", "grey"]
        mid = self.size // 2

        for j in [5, 11]:
            for i in range(width):
                self.grid.set(i, j, Wall())

            for i in range(height):
                self.grid.set(j, i, Wall())

        room_options = [
            ((5, mid), (1, 6), (4, 5)),
            ((11, mid), (12, 6), (4, 5)),
            ((mid, 5), (6, 1), (5, 4)),
            ((mid, 11), (6, 12), (5, 4))
        ]

        random.shuffle(room_options)

        goal_room = room_options[0]
        key_room = room_options[1]
        other_rooms = room_options[2:]

        target_color = colors.pop(random.randrange(len(colors)))

        door_pos, zone_top, zone_size = goal_room
        self.grid.set(door_pos[0], door_pos[1], Door(target_color, is_locked=True))
        self.place_obj(Goal(), top=zone_top, size=zone_size)

        door_pos, zone_top, zone_size = key_room
        door_color = colors.pop(random.randrange(len(colors)))
        self.grid.set(door_pos[0], door_pos[1], Door(door_color, is_locked=False))
        self.place_obj(Key(target_color), top=zone_top, size=zone_size)

        for room in other_rooms:
            door_pos = room[0]
            door_color = colors.pop(random.randrange(len(colors)))
            self.grid.set(door_pos[0], door_pos[1], Door(door_color, is_locked=False))

        self.place_agent(top=(6, 6), size=(5, 5))
        self.mission = "goal"

    def step(self, action):
        obs, _, terminated, truncated, info = super().step(action)

        if terminated:
            reward = CURRICULUM_REWARDS["goal"]
        else:
            reward = CURRICULUM_REWARDS["normal"]

        return obs, reward, terminated, truncated, info


def make_color_env():
    env = ColorEnv(render_mode="rgb_array")
    env = ConvWrapper(env)
    return env