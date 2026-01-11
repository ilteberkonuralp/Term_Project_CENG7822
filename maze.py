# -*- coding: utf-8 -*-
"""
maze.py - Customizable Number of Checkpoints
"""

import gym
import numpy as np
from gym import spaces
from matplotlib import pyplot as plt
import collections

class Maze(gym.Env):
    def __init__(self, grid_size=10, seed=42, sequential_mode=False, num_checkpoints=4):
        super(Maze, self).__init__()
        
        self.grid_size = grid_size
        self.rng = np.random.RandomState(seed)
        self.sequential_mode = sequential_mode
        self.num_checkpoints = num_checkpoints  # <--- NEW PARAMETER
        
        self.actions = [(-1, 0), (1, 0), (0, 1), (0, -1)]
        self.action_space = spaces.Discrete(len(self.actions))
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.grid_size, self.grid_size), dtype=np.int32)
        
        # --- DEFINING THE PATH ---
        self.start = (self.grid_size - 1, 0)      # Bottom-Left
        self.final_goal = (0, self.grid_size - 1) # Top-Right
        
        # --- DYNAMIC CHECKPOINTS ---
        # Calculate N evenly spaced points along the diagonal
        self.waypoints_config = []
        
        # If num_checkpoints is 0, we just go Start -> Goal
        if self.num_checkpoints > 0:
            for i in range(1, self.num_checkpoints + 1):
                # Interpolation from Start to Goal
                alpha = i / (self.num_checkpoints + 1)
                r = int((self.grid_size - 1) * (1 - alpha))
                c = int((self.grid_size - 1) * alpha)
                self.waypoints_config.append((r, c))
            
        # Generate the map ONCE.
        self.maze = self._generate_guaranteed_maze()
        
        self.current_waypoints = []
        self.waypoint_index = 0
        
        self.reset()

    @property
    def goal(self):
        if not self.sequential_mode:
            return self.final_goal
        if self.waypoint_index < len(self.current_waypoints):
            return self.current_waypoints[self.waypoint_index]
        return self.final_goal

    def reset(self):
        self.state = self.start
        self.current_waypoints = list(self.waypoints_config)
        self.waypoint_index = 0
        self.done = False
        return self.state

    def _generate_guaranteed_maze(self):
        """
        Generates a map and strictly validates the full chain:
        Start -> CP1 -> CP2 -> ... -> CP_N -> Final Goal.
        """
        while True:
            map_grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)
            
            # Random walls (Density 30%)
            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    if self.rng.rand() < 0.3: 
                        map_grid[i, j] = 1
            
            # 1. Clear Walls at Critical Points
            map_grid[self.start] = 0
            map_grid[self.final_goal] = 0
            for wp in self.waypoints_config:
                map_grid[wp] = 0
            
            # 2. Strict Chain Validation
            path_points = [self.start] + self.waypoints_config + [self.final_goal]
            all_segments_valid = True
            
            for i in range(len(path_points) - 1):
                p1 = path_points[i]
                p2 = path_points[i+1]
                if not self._is_solvable(map_grid, p1, p2):
                    all_segments_valid = False
                    break 
            
            if all_segments_valid:
                return map_grid

    def _is_solvable(self, grid, start, goal):
        queue = collections.deque([start])
        visited = set([start])
        while queue:
            current = queue.popleft()
            if current == goal: return True
            cx, cy = current
            for dx, dy in self.actions:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                    if grid[nx, ny] == 0 and (nx, ny) not in visited:
                        visited.add((nx, ny))
                        queue.append((nx, ny))
        return False

    def step(self, action):
        x, y = self.state
        dx, dy = self.actions[action]
        nx, ny = x + dx, y + dy
        
        if nx < 0 or nx >= self.grid_size or ny < 0 or ny >= self.grid_size:
            reward = -0.5
            next_state = (x, y) 
        elif self.maze[nx, ny] == 1:
            reward = -0.5
            next_state = (x, y) 
        else:
            next_state = (nx, ny)
            reward = -0.1 

            current_target = self.goal
            
            if next_state == current_target:
                if self.sequential_mode and self.waypoint_index < len(self.current_waypoints):
                    reward = 5.0 
                    self.waypoint_index += 1 
                else:
                    reward = 100.0
                    self.done = True
        
        self.state = next_state
        return self.state, reward, self.done, {}

    def render(self, episode=0, t=0, sub_goal=None):
        view_map = np.copy(self.maze).astype(float)
        view_map[self.final_goal] = 0.3 
        view_map[self.state] = 0.8 
        
        if self.sequential_mode:
            active_target = self.goal
            view_map[active_target] = 0.6 
        else:
            view_map[self.final_goal] = 0.6

        plt.clf()
        
        if self.sequential_mode and self.waypoint_index < len(self.current_waypoints):
            status = f"Target: CP {self.waypoint_index + 1}/{len(self.current_waypoints)} at {self.current_waypoints[self.waypoint_index]}"
        else:
            status = "Target: FINAL GOAL"
            
        plt.title(f'Ep: {episode} | {status}')
        plt.imshow(view_map, cmap='viridis')
        plt.xticks([]) 
        plt.yticks([])
        plt.ion()
        plt.draw()
        plt.pause(0.001)

    def get_valid_indices(self):
        valid_coords = np.argwhere(self.maze == 0)
        return [int(self.grid_size * x + y) for x, y in valid_coords]