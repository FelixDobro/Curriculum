import numpy as np
from gymnasium import ObservationWrapper, spaces
from minigrid.core.world_object import Goal, Wall
from minigrid.core.constants import COLORS
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.minigrid_env import MiniGridEnv
import random
from envs.wrappers import ConvWrapper

from config import CURRICULUM_STEPS, CURRICULUM_REWARDS

def _gen_mission():
        return "navigate clutter"
class Maze(MiniGridEnv):
    
    def __init__(self, size=40, n_obstacles=700, max_steps=100, **kwargs):
        self.n_obstacles = n_obstacles
        mission_space = MissionSpace(mission_func=_gen_mission)
        super().__init__(
            mission_space=mission_space,
            width=size,
            height=size,
            max_steps=max_steps,
            **kwargs,
        )

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)

        x, y = self.place_agent()
        goal_pos = self.place_obj(Goal())


        #dfs with backtrackting to generate a random path from start to goal

        seen = set([(x,y)])
        stack = [((x, y), [(x,y)])]
  
        seen.add((x, y))
        safe_path = None
        while stack:
            
            (x,y), current_path = stack.pop()
            
            if (x,y) == goal_pos:
                safe_path = set(current_path)
                break

            neighbors = [
                (x+1, y),
                (x, y+1),
                (x-1, y),
                (x, y-1)
            ]
            random.shuffle(neighbors)
            for nx, ny in neighbors:
                if not (nx, ny) in seen:
                    if 0 < nx <  width - 1 and 0 < ny < height -1:
                        seen.add((nx, ny))
                        stack.append(((nx,ny), current_path + [(nx,ny)]))
                    
            
        obstacles_placed = 0

        for _ in range(5000):
            if obstacles_placed == self.n_obstacles: break
            x = random.randint(1, width - 2)
            y = random.randint(1, height - 2)
            
            # Nur setzen, wenn Feld leer UND nicht auf dem Pfad
            if (x, y) not in safe_path and self.grid.get(x, y) is None:
                self.grid.set(x, y, Wall())
                obstacles_placed += 1
    
        '''for x, y in safe_path:
            # Check 1: Ist dort das Ziel?
            cell = self.grid.get(x, y)
            if cell and cell.type == "goal":
                continue # Nicht übermalen!
            
            # Check 2: Ist dort der Agent?
            if (x, y) == tuple(self.agent_pos):
                continue # Nicht übermalen!

            # Jetzt erst die Markierung setzen
            self.grid.set(x, y, Wall(color="red"))'''

        self.mission = "navigate clutter"



    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        if terminated:
            reward = CURRICULUM_REWARDS["goal"]
            info["success"] = True
        else:
            info["success"] = False
            reward = CURRICULUM_REWARDS["normal"]
        return obs, reward, terminated, truncated, info


    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        info["success"] = False 

        return obs, info

def make_maze_diff0():
    env = Maze(size=8, n_obstacles=9, max_steps=CURRICULUM_STEPS["maze_0"], render_mode="rgb_array")
    env = ConvWrapper(env)
    return env

def make_maze_diff1():
    env = Maze(size=11, n_obstacles=30, max_steps=CURRICULUM_STEPS["maze_1"], render_mode="rgb_array")
    env = ConvWrapper(env)
    return env

def make_maze_diff2():
    env = Maze(size=13, n_obstacles=40, max_steps=CURRICULUM_STEPS["maze_2"], render_mode="rgb_array")
    env = ConvWrapper(env)
    return env

def make_maze_diff3():
    env = Maze(size=15, n_obstacles=60, max_steps=CURRICULUM_STEPS["maze_3"], render_mode="rgb_array")
    env = ConvWrapper(env)
    return env

def make_maze_diff4():
    env = Maze(size=17, n_obstacles=88, max_steps=CURRICULUM_STEPS["maze_4"], render_mode="rgb_array")
    env = ConvWrapper(env)
    return env

def make_maze_diff5():
    env = Maze(size=20, n_obstacles=(25**2)//2.75, max_steps=CURRICULUM_STEPS["maze_5"], render_mode="rgb_array")
    env = ConvWrapper(env)
    return env
    