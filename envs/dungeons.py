import random

from minigrid.core.grid import Grid
from minigrid.core.world_object import *
from minigrid.minigrid_env import MiniGridEnv
from minigrid.core.mission import MissionSpace
import numpy as np

from config import CURRICULUM_REWARDS, CURRICULUM_STEPS
from envs.wrappers import ConvWrapper


def maze_backtrack(start_x, start_y, x_end, y_end, grid, destination):
    seen = set([(start_x, start_y)])
    stack = [((start_x, start_y), [(start_x, start_y)])]

    safe_path = []

    while stack:
        (x, y), current_path = stack.pop()
        
        if (x, y) == destination:
            safe_path = current_path
            break
            
        neighbors = [(x + 1, y), (x, y + 1), (x - 1, y), (x, y - 1)]
        random.shuffle(neighbors)
        
        for nx, ny in neighbors:
            # Jetzt darf er sich in der ganzen Box frei bewegen
            if start_x <= nx <= x_end and start_y <= ny <= y_end:
                if (nx, ny) not in seen:
                    seen.add((nx, ny))
                    stack.append(((nx, ny), current_path + [(nx, ny)]))
                
    # Pfad freiräumen (Bulldozer-Modus!)
    for x, y in safe_path:
        # GANZ WICHTIG: Den Schlüssel (destination) nicht plattmachen!
        if (x, y) != destination:
            grid[x, y] = "."
       

class Room:

    def generate():
        pass

class BasicRoom(Room):

    def __init__(self):

        self.room_string = """#######
        #.....#
        #.....+
        #.....#
        #######
        """.strip()
        
    def generate(self):
        grid = np.array([list(row.strip()) for row in self.room_string.splitlines()])
      
        random_row = random.randint(1, grid.shape[0] - 2)
        random_col = random.randint(1, grid.shape[1] - 2)
        grid[random_row, random_col] = "A"
        return grid


class KeyRoom(Room): 
    def __init__(self):

        self.room_string = """
        #######
        #.....#
        #.....#
        +.....L
        #.....#
        #.....#
        #######
        """.strip()
        
    def generate(self):
        grid = np.array([list(row.strip()) for row in self.room_string.splitlines()])
       
        agent_x = random.randint(3, 4)
        agent_y = random.randint(1,2)
        
        key_x = random.choice([1,5])
        key_y = random.randint(1,5)

        grid[agent_x, agent_y] = "A"
        grid[key_x, key_y] = "K"

        return grid



class LockedIn(Room): 
    def __init__(self):

        self.room_string = """
        #########################################
        #.....#..........................#......#
        #.....D..........................#......#
        #.....#..........................##+##+##
        #LLLLL#..........................#......#
        #.....#..........................#......#
        #LLLLL#..........................##+##+##
        #.....#..........................#......#
        #LLLLL#..........................#......#
        #.....#..........................#......#
        #.....#..........................#......#
        #########################################
        """.strip()
        
    def generate(self):
        grid = np.array([list(row.strip()) for row in self.room_string.splitlines()])
       
        agent_x = random.randint(1, 2)
        grid[agent_x, random.randint(1,4)] = "A"

        lava_gap_x = [4,6,8]
        lava_gap_y = [random.randint(1,5),random.randint(1,5),random.randint(1,5)]
        grid[lava_gap_x, lava_gap_y] = "."
        random_key_x, random_key_y = random.randint(9,10), random.randint(1,5)
        grid[random_key_x, random_key_y] = "K"
     
        random_obstacles_y = np.random.randint(8, 32 , (130))
        random_obstacles_x = np.random.randint(1,10, (130))
        grid[random_obstacles_x, random_obstacles_y] = "#"

        random_key_maze_y = random.randint(12, 30)
        random_key_maze_x = random.randint(1,10)
        
        random_door_x = random.randint(2, 10)
        door_y = 33

        grid[random_door_x, door_y] = "D"
        maze_backtrack(2, 7, 10, 32, grid, destination=(random_key_maze_x, random_key_maze_y))
        maze_backtrack(random_key_maze_x, random_key_maze_y, random_door_x, door_y -1 , grid, destination=(random_door_x, door_y -1))
        grid[random_key_maze_x, random_key_maze_y] = "K"
        
        goal_x = random.choice([1,2,4,5,7,8,9,10])
        goal_y = random.randint(34,39)
        
        grid[goal_x, goal_y] = "G"
        return grid


class Dungeon(MiniGridEnv):

    def __init__(self):
        super().__init__(
            mission_space=MissionSpace(mission_func=lambda:"Goal"),
            render_mode="rgb_array",
            height=10,
            width=10,
            max_steps=CURRICULUM_STEPS["dungeon_1"]
        ) 

        self.room_choices = [BasicRoom(), KeyRoom()]
        
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

        
    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        info["success"] = False 

        return obs, info
            
    def _gen_grid(self,_,__):

        grid = LockedIn().generate()
        flip = random.randint(0,3)
        for _ in range(flip):
            grid = np.rot90(grid)
        self.height, self.width = grid.shape
        self.grid = Grid(self.width, self.height)

        for y, row in enumerate(grid):
            for x, tile in enumerate(row):
                if tile == ".":
                    continue
                elif tile == "#":
                    self.grid.set(x,y, Wall())
                elif tile == "+":
                    self.grid.set(x,y, Door(color="red"))
                elif tile == "D":
                    self.grid.set(x, y, Door(color="blue", is_locked=True))
                elif tile == "K":
                    self.grid.set(x, y, Key(color="blue"))

                elif tile == "L":
                    self.grid.set(x,y, Lava())

                elif tile == "A":
                    self.agent_pos = (x, y)
                    self.agent_dir = random.choice([0,1,2,3])
                    self.grid.set(x, y, None)
                elif tile == "G":
                    self.grid.set(x,y, Goal())
                else:
                    continue                   


def make_dungeon_1():
    env = Dungeon()
    env = ConvWrapper(env)
    return env