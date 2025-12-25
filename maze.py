# -*- coding: utf-8 -*-
"""
Spyder Editor

Fixed 10x10 Maze Environment
"""

import gym
import numpy as np
from gym import spaces
from matplotlib import pyplot as plt
import collections

class Maze(gym.Env):
    def __init__(self):
        super(Maze, self).__init__()
        
        # --- Configuration ---
        self.grid_size = 10  # <--- CHANGED FROM 30 TO 10
        self.obstacle_density = 0.25 
        
        self.actions = [(-1, 0), (1, 0), (0, 1), (0, -1)] # Up, Down, Right, Left
        self.action_space = spaces.Discrete(len(self.actions))
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.grid_size, self.grid_size), dtype=np.int32)
        
        # Fixed Start and Goal
        self.start = (self.grid_size - 1, 0) # Bottom-Left
        self.goal = (0, self.grid_size - 1)  # Top-Right
        
        # --- GENERATE MAP ONCE (FIXED) ---
        # We set a seed so the 'random' walls are the same every time
        np.random.seed(42) 
        self.maze = self._generate_fixed_map()
        
        self.state = None
        self.reset()

    def reset(self):
        """
        Resets the agent to the start position. 
        The map does NOT change.
        """
        self.state = self.start
        self.done = False
        return self.state

    def _generate_fixed_map(self):
        """Generates a valid map once."""
        while True:
            # 0 = Empty, 1 = Wall
            map_grid = np.zeros((self.grid_size, self.grid_size))
            
            # Place walls based on seed
            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    if np.random.rand() < self.obstacle_density:
                        map_grid[i, j] = 1
            
            # Ensure Start and Goal are empty
            map_grid[self.start] = 0
            map_grid[self.goal] = 0
            
            # If this specific seed produced a solvable map, keep it.
            if self._is_solvable(map_grid):
                return map_grid
            else:
                # If seed 42 was bad, try the next one until we find a fixed valid map
                # (This loop only runs once during initialization)
                pass

    def _is_solvable(self, maze):
        """Uses BFS to check if there is a path from Start to Goal."""
        queue = collections.deque([self.start])
        visited = set([self.start])
        
        while queue:
            current = queue.popleft()
            if current == self.goal:
                return True
            
            x, y = current
            for dx, dy in self.actions:
                nx, ny = x + dx, y + dy
                
                # Check bounds
                if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                    # Check if not wall and not visited
                    if maze[nx, ny] == 0 and (nx, ny) not in visited:
                        visited.add((nx, ny))
                        queue.append((nx, ny))
        return False

    def step(self, action):
        x = self.state[0] + self.actions[action][0]
        y = self.state[1] + self.actions[action][1]
        
        # Check Bounds
        if x < 0 or x >= self.grid_size or y < 0 or y >= self.grid_size:
            reward = -1
        # Check Obstacles
        elif self.maze[x, y] == 1:
            reward = -1
        # Check Goal
        elif (x, y) == self.goal:
            self.state = (x, y)
            reward = 0
            self.done = True
            return self.state, reward, self.done, {}
        # Normal Move
        else:
            self.state = (x, y)
            reward = -1
            
        return self.state, reward, self.done, {}

    def render(self, episode=0, t=0, mode='human'):
        view_map = np.copy(self.maze)
        view_map[self.goal] = 0.5 
        view_map[self.state] = 0.8 
        
        plt.clf()
        plt.title(f'Episode: {episode} Step: {t}')
        plt.imshow(view_map, cmap='viridis')
        plt.xticks([]) 
        plt.yticks([])
        plt.ion()
        plt.draw()
        plt.pause(0.001)

if __name__=='__main__':
    env = Maze()
    # Reduced figure size slightly since the grid is smaller
    plt.figure(figsize=(5,5)) 
    
    for i_episode in range(1):
        state = env.reset()
        print(f'--- Starting Episode {i_episode} ---')
        
        for t in range(200):
            action = env.action_space.sample()
            state, reward, done, info = env.step(action)
            env.render(i_episode, t)
            if done:
                print("Goal reached!")
                break