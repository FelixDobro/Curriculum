from __future__ import annotations
import random
import numpy as np
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Goal, Key, Door, Wall
from minigrid.minigrid_env import MiniGridEnv
from config import CURRICULUM_REWARDS, CURRICULUM_STEPS
from envs.wrappers import ConvWrapper

class FullDynamicRoomEnv(MiniGridEnv):
    def __init__(
            self,
            max_steps=100,
            room_grid_size=4,      # 3 = 3x3 (9 Räume), 4 = 4x4 (16 Räume), etc.
            num_locked_doors=4,    # Wie viele Türen sollen verschlossen sein?
            room_size=5,           # Größe eines einzelnen Raums (5x5 Pixel)
            render_mode="rgb_array",
            **kwargs,
    ):
        self.room_grid_size = room_grid_size
        self.num_locked_target = num_locked_doors
        self.room_size = room_size
        
        # Grid-Größe berechnen: 
        # (Anzahl Räume * Raumgröße) + Trennwände + Außenwände
        self.grid_pixel_size = room_grid_size * (room_size + 1) + 1

        # Max Steps anpassen: Mehr Räume = Mehr Zeit nötig
        total_rooms = room_grid_size * room_grid_size
       

        mission_space = MissionSpace(mission_func=lambda: "traverse the rooms to find the goal")

        super().__init__(
            mission_space=mission_space,
            width=self.grid_pixel_size,
            height=self.grid_pixel_size,
            max_steps=max_steps,
            render_mode=render_mode,
            see_through_walls=False,
            **kwargs,
        )

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)

        # 1. Das Gittergerüst bauen (Alle Wände ziehen)
        for i in range(1, self.room_grid_size):
            pos = i * (self.room_size + 1)
            # Vertikale Trennwände
            for y in range(height): self.grid.set(pos, y, Wall())
            # Horizontale Trennwände
            for x in range(width): self.grid.set(x, pos, Wall())

        # Hilfsfunktion: Gibt Koordinaten (top_left, size) für Raum (x,y) zurück
        def get_room_coords(rx, ry):
            x = rx * (self.room_size + 1) + 1
            y = ry * (self.room_size + 1) + 1
            return (x, y), (self.room_size, self.room_size)

        # 2. Alle Räume definieren
        # Wir nutzen jetzt IMMER alle Räume des Grids, keine Lücken mehr.
        all_rooms = [(x, y) for x in range(self.room_grid_size) for y in range(self.room_grid_size)]
        
        # Start zufällig wählen
        start_room = random.choice(all_rooms)
        visited_rooms = {start_room} # Set für schnelle Suche
        
        # Agent platzieren
        top, size = get_room_coords(*start_room)
        self.place_agent(top=top, size=size)

        colors = ["red", "green", "blue", "purple", "yellow", "grey"]
        available_colors = colors * 2 # Farben verdoppeln falls wir > 6 Schlösser haben
        locks_placed = 0
        
        # 3. Expansion (Prim's Algorithmus / Region Growing)
        # Wir machen solange weiter, bis ALLE Räume besucht sind
        while len(visited_rooms) < len(all_rooms):
            
            # Finde alle möglichen Verbindungen von "besucht" zu "unbesucht"
            candidates = []
            for r in visited_rooms:
                rx, ry = r
                # Nachbarn checken (Oben, Unten, Links, Rechts)
                neighbors = [(rx+1, ry), (rx-1, ry), (rx, ry+1), (rx, ry-1)]
                
                for n in neighbors:
                    if n in all_rooms and n not in visited_rooms:
                        candidates.append((n, r)) # (Ziel, Start)
            
            if not candidates:
                break
            
            # Wähle eine zufällige Verbindung -> Das garantiert zufällige Labyrinthe
            next_room, source_room = random.choice(candidates)
            
            # --- TÜR PLATZIEREN ---
            nx, ny = next_room
            sx, sy = source_room
            
            # Position der Tür berechnen
            if nx != sx: # Vertikal getrennt
                wall_x = max(nx, sx) * (self.room_size + 1)
                # Zufällige y-Position in der Wand, nicht immer strikt Mitte
                # Aber Mitte ist sicherer für Pathfinding
                wall_y = ny * (self.room_size + 1) + 1 + random.randint(0, self.room_size - 1)
                door_pos = (wall_x, wall_y)
            else: # Horizontal getrennt
                wall_x = nx * (self.room_size + 1) + 1 + random.randint(0, self.room_size - 1)
                wall_y = max(ny, sy) * (self.room_size + 1)
                door_pos = (wall_x, wall_y)

            # --- SCHLOSS LOGIK ---
            # Wir verschließen die Tür, wenn wir noch Budget haben UND Farben übrig sind
            should_lock = False
            if locks_placed < self.num_locked_target and available_colors:
                # 70% Chance zu verschließen, damit nicht alle Schlösser sofort am Anfang kommen
                if random.random() < 0.7: 
                    should_lock = True
            
            # Fallback: Wenn wir fast am Ende sind und noch Schlösser fehlen, erzwingen wir es
            rooms_left = len(all_rooms) - len(visited_rooms)
            locks_left = self.num_locked_target - locks_placed
            if locks_left >= rooms_left and available_colors:
                should_lock = True

            if should_lock:
                c = available_colors.pop(0)
                self.grid.set(door_pos[0], door_pos[1], Door(c, is_locked=True))
                
                # SCHLÜSSEL PLATZIEREN
                # Der Schlüssel muss in einem BEREITS besuchten Raum liegen (Backtracking garantiert)
                key_room = random.choice(list(visited_rooms))
                k_top, k_size = get_room_coords(*key_room)
                self.place_obj(Key(c), top=k_top, size=k_size, max_tries=100)
                
                locks_placed += 1
            else:
                # Offene Tür (Farbe grey ist neutral)
                self.grid.set(door_pos[0], door_pos[1], Door("grey", is_locked=False))

            visited_rooms.add(next_room)
            self.last_room = next_room # Merken für Goal

        # 4. Goal im allerletzten Raum platzieren (garantiert längster Pfad)
        g_top, g_size = get_room_coords(*self.last_room)
        self.place_obj(Goal(), top=g_top, size=g_size)

    def step(self, action):
        obs, _, terminated, truncated, info = super().step(action)
        
        reward = 0
        if terminated:
            reward = CURRICULUM_REWARDS["goal"]
        # Kleiner Schritt-Penalty, damit er nicht trödelt (optional)
        # else:
        #    reward = -0.01

        return obs, reward, terminated, truncated, info

def difficulty_1(size=2, locks=1, max_steps=CURRICULUM_STEPS["difficulty1"]):

    env = FullDynamicRoomEnv(room_grid_size=size, num_locked_doors=locks, max_steps=max_steps)
    env = ConvWrapper(env)
    return env

def difficulty_2(size=2, locks=2, max_steps=CURRICULUM_STEPS["difficulty2"]):

    env = FullDynamicRoomEnv(room_grid_size=size, num_locked_doors=locks, max_steps=max_steps)
    env = ConvWrapper(env)
    return env

def difficulty_3(size=3, locks=1, max_steps=CURRICULUM_STEPS["difficulty3"]):

    env = FullDynamicRoomEnv(room_grid_size=size, num_locked_doors=locks, max_steps=max_steps)
    env = ConvWrapper(env)
    return env

def difficulty_4(size=3, locks=2, max_steps=CURRICULUM_STEPS["difficulty4"]):

    env = FullDynamicRoomEnv(room_grid_size=size, num_locked_doors=locks, max_steps=max_steps)
    env = ConvWrapper(env)
    return env


def difficulty_5(size=3, locks=3, max_steps=CURRICULUM_STEPS["difficulty5"]):

    env = FullDynamicRoomEnv(room_grid_size=size, num_locked_doors=locks, max_steps=max_steps)
    env = ConvWrapper(env)
    return env

def difficulty_6(size=3, locks=5, max_steps=CURRICULUM_STEPS["difficulty6"]):

    env = FullDynamicRoomEnv(room_grid_size=size, num_locked_doors=locks, max_steps=max_steps)
    env = ConvWrapper(env)
    return env

def difficulty_7(size=3, locks=7, max_steps=CURRICULUM_STEPS["difficulty7"]):

    env = FullDynamicRoomEnv(room_grid_size=size, num_locked_doors=locks, max_steps=max_steps)
    env = ConvWrapper(env)
    return env

