#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import time
from hetmarl.envs.gridworld.gym_minigrid.minigrid import *
from .multiroom import *
import cv2
import random
import copy
import numpy as np

from hetmarl.envs.gridworld.frontier.apf import APF
from hetmarl.envs.gridworld.frontier.utility import utility_goal
from hetmarl.envs.gridworld.frontier.rrt import rrt_goal
from hetmarl.envs.gridworld.frontier.nearest import nearest_goal
from hetmarl.envs.gridworld.frontier.voronoi import voronoi_goal

from hetmarl.envs.gridworld.gym_minigrid.register import register

from hetmarl.utils import astar
from hetmarl.utils.util import get_connected_agents
from hetmarl.utils import dstarlite, plotting, env
import pyastar2d


def l1distance(x, y):
    return abs(x[0] - y[0]) + abs(x[1] - y[1])

def euclideandistance(x, y):
    return math.sqrt((x[0] - y[0])**2 + (x[1] - y[1])**2)


class SearchAndRescueEnv(MultiRoomEnv):
    """
    Classic 4 rooms gridworld environment.
    Can specify agent and goal position, if not it set at random.
    """

    def __init__(
        self,
        grid_size,
        max_steps,
        local_step_num,
        agent_view_size,
        num_obstacles,
        agent_types_list,
        num_agents=2,
        n_agent_types=2,
        agent_pos=None,
        goal_pos=None,
        use_full_comm=False,
        use_partial_comm=False,
        use_orientation=False,
        use_same_location=True,
        use_time_penalty=False,
        use_energy_penalty=False,
        use_intrinsic_reward=False,
        use_agent_obstacle = False,
        algorithm_name = 'amat',
        trajectory_forget_rate = 0.8,
        com_sigma = 5,
        block_doors = False,
        block_chance = 0.2,
        detect_traces = False,
        action_size = 5
    ):
        self.grid_size = grid_size
        self._agent_default_pos = agent_pos
        self._goal_default_pos = goal_pos
        self.door_size = 1
        self.max_steps = max_steps
        self.use_same_location = use_same_location
        self.use_time_penalty = use_time_penalty
        self.use_energy_penalty = use_energy_penalty
        self.use_intrinsic_reward = use_intrinsic_reward

        self.use_full_comm = use_full_comm
        self.use_partial_comm = use_partial_comm
        self.use_agent_obstacle = use_agent_obstacle
        self.maxNum = grid_size // 5 + 1
        self.minNum = grid_size // 8
        self.trajectory_forget_rate = trajectory_forget_rate
        self.n_agent_types = n_agent_types
        self.original_agent_types_list = agent_types_list
        self.com_sigma = com_sigma
        self.block_doors = block_doors
        self.block_chance = block_chance
        self.detect_traces = detect_traces
        self.action_size = action_size

        if num_obstacles <= grid_size/2 + 1:
            self.num_obstacles = int(num_obstacles)
        else:
            self.num_obstacles = int(grid_size**2/40)

        super().__init__(minNumRooms=4,
                         maxNumRooms=7,
                         maxRoomSize=8,
                         grid_size=grid_size,
                         max_steps=max_steps,
                         num_agents=num_agents,
                         agent_view_size=agent_view_size,
                         use_full_comm=use_full_comm,
                         use_partial_comm=use_partial_comm,
                         use_orientation=use_orientation,
                         algorithm_name=algorithm_name
                         )
        self.target_ratio = 0.90
        self.merge_ratio = 0
        self.merge_reward = 0
        self.episode_no = 0
        self.agent_exploration_reward = np.zeros((num_agents))
        self.max_steps = max_steps
        self.local_step_num = local_step_num
    def overall_gen_grid(self, width, height):

        # Create the grid
        self.grid = Grid(width, height)
        
        # Generate the surrounding walls
        self.grid.horz_wall(0, 0)
        self.grid.horz_wall(0, height - 1)
        self.grid.vert_wall(0, 0)
        self.grid.vert_wall(width - 1, 0)

        w = self._rand_int(self.minNum, self.maxNum)
        h = self._rand_int(self.minNum, self.maxNum)

        room_w = width // w
        room_h = height // h
        
        self.doorways = set()
        self.doorways_adjacent_cells = set()
        self.original_rubble_set = set()
        # For each row of rooms
        for j in range(0, h):

            # For each column
            for i in range(0, w):
                xL = i * room_w
                yT = j * room_h
                xR = xL + room_w
                yB = yT + room_h

                # Bottom wall and door
                if i + 1 < w:
                
                    if self._rand_float(0, 1) < 0.1: # and i != 0 and i != w-2:
                        pass
                    else:
                        if j == h - 1:
                            self.grid.vert_wall(xR, yT)
                        else:
                            self.grid.vert_wall(xR, yT, room_h)
                        pos = (xR, self._rand_int(yT + 1, yB - 1))

                        for s in range(self.door_size):
                            self.grid.set(*pos, Door(color='grey',is_open=True))
                            self.doorways.add(tuple(pos))
                            self.doorways_adjacent_cells = self.doorways_adjacent_cells.union(self.adjacent_cells(*pos))
                            if self.block_doors and self._rand_float(0, 1) < self.block_chance: #randomly block some doorways
                                # self.grid.set(pos[0],pos[1], Rubble())
                                # self.grid.set(pos[0]+self._rand_int(-1,1),pos[1], Rubble())
                                rubble_pos = self.place_obj(Rubble(), top = (pos[0]-1, pos[1]-1), size = (2,2), reject_fn=accept_near_doors)
                                self.original_rubble_set.add(tuple(rubble_pos))
                            pos = (pos[0], pos[1] + 1)

                # Bottom wall and door
                if j + 1 < h:

                    if self._rand_float(0, 1) < 0.1:#  and j != 0 and j != h-2:
                        pass
                    else:
                        if i == w - 1:
                            self.grid.horz_wall(xL, yB)
                        else:
                            self.grid.horz_wall(xL, yB, room_w)
                        pos = (self._rand_int(xL + 1, xR - 1), yB)
                        self.grid.set(*pos, Door(color='grey',is_open=True))
                        self.doorways.add(tuple(pos))
                        self.doorways_adjacent_cells = self.doorways_adjacent_cells.union(self.adjacent_cells(*pos))
                        if self.block_doors and self._rand_float(0, 1) < self.block_chance: #randomly block some doorways
                            # self.grid.set(pos[0],pos[1], Rubble())
                            # self.grid.set(pos[0],pos[1]+self._rand_int(-1,1), Rubble())
                            rubble_pos = self.place_obj(Rubble(), top = (pos[0]-1, pos[1]-1), size = (2,2), reject_fn=accept_near_doors)
                            self.original_rubble_set.add(tuple(rubble_pos))
        # print("original rubble set is: ", self.original_rubble_set)
        self.doorways_adjacent_cells = self.doorways_adjacent_cells.union(self.doorways)
        
        if self._agent_default_pos is not None and len(self._agent_default_pos) > 0:
            self.agent_pos = self._agent_default_pos
            # for i in range(self.num_agents):
            #     self.grid.set(*self._agent_default_pos[i], None)
            self.agent_dir = [self._rand_int(0, 4) for i in range(
                self.num_agents)]  # assuming random start direction
        else: # Randomize the player start position and orientation
            # self.place_agent(use_same_location=self.use_same_location)
            #only putting the agents near corners
            starting_corner_x = self._rand_int(0, 2)
            starting_corner_y = self._rand_int(0, 2)
            self.place_agent(use_same_location=self.use_same_location,top=(1+starting_corner_x*(width-5),1+starting_corner_y*(height-5)),size=(4,4))

        # place random obstacles, but do not block doorways
        self.obstacles = []
        for i_obst in range(self.num_obstacles):
            self.obstacles.append(Obstacle())
            pos = self.place_obj(self.obstacles[i_obst], max_tries=100, reject_fn=reject_near_doors)
        
        # form an initial list of occupied space
        obs_list = []
        for x in range(width):
            for y in range(height):
                if self.grid.get(x, y) is not None and (self.grid.get(x, y).type == 'obstacle' or self.grid.get(x, y).type == 'wall' \
                                                        or self.grid.get(x, y).type == 'rubble'):
                    obs_list.append((x, y))
        
        def reject_bad_target_locations(self, pos):
            x, y = pos
            if x >= starting_corner_x*(width-8) and x < (starting_corner_x)*(width-8) + 8 \
                and y >= starting_corner_y*(height-8) and y < (starting_corner_y)*(height-8) + 8:
                return True 
            if tuple(pos) in self.doorways_adjacent_cells:
                return True
            # num_tries = 0
            # while num_tries < 100:
            #     test_pos = np.array((self._rand_int(0,  self.grid.width),
            #                         self._rand_int(0, self.grid.height)))
            #     path_planner = astar.AStar(obs_list, tuple(test_pos), tuple(pos), "manhattan")
            #     path, _ = path_planner.searching()
            #     if len(path) < 5: #target is trapped and unreachable
            #         continue
            #     else:
            #         return False
            #     num_tries += 1
            # return True
            return False

        def reject_far_from_target(self, pos):
            #reject if far from target or unreachable
            path_planner = astar.AStar(obs_list, tuple(pos), tuple(self.target_pos), "manhattan")
            path, _ = path_planner.searching()
            if len(path) < 5 or len(path) > 14 : # len(path) == 1 is unreachable
                return True

        # place a target randomly in the environment
        if self._goal_default_pos is not None:
            target = Goal()
            self.put_obj(target, *self._goal_default_pos)
            target.init_pos, target.cur_pos = self._goal_default_pos
        else:
            self.target_pos = self.place_obj(Goal(), reject_fn = reject_bad_target_locations)
            target_adjacent_cells = self.adjacent_cells(*self.target_pos, surround = True)
            for cell in target_adjacent_cells: #place traces on target adjacent cells
                self.place_obj(Trace(), top = cell, size = (1,1))
            # self.initial_trace = self.place_obj(Trace(), reject_fn = reject_far_from_target)
        
        # if self.initial_trace[0] != -1:
        #     # Generate a path for the target to somewhere close
        #     path_planner = astar.AStar(obs_list, tuple(self.initial_trace), tuple(self.target_pos), "manhattan")
        #     path, _ = path_planner.searching()
        #     for i in range(1, len(path)):
        #         if self._rand_float(0, 1) < 0.75:
        #             self.grid.set(*path[i], Trace())


        self.mission = 'Reach the goal'
    
    def astar_path(self, start, goal):
        obs_list = []
        for x in range(self.width):
            for y in range(self.height):
                if self.grid.get(x, y) is not None and (self.grid.get(x, y).type == 'obstacle' or self.grid.get(x, y).type == 'wall' \
                                                        or self.grid.get(x, y).type == 'rubble'):
                    obs_list.append((x, y))
        path_planner = astar.AStar(obs_list, tuple(start), tuple(goal), "manhattan")
        path, _ = path_planner.searching()
        path_length = len(path)
        return path, path_length

    def set_target_each_map(self, agent_id):
        self.target_each_map[agent_id] = np.zeros((self.width + 2*self.agent_view_size, self.height + 2*self.agent_view_size))
        self.global_target_each_map[agent_id] = np.zeros((self.width + 2*self.agent_view_size, self.height + 2*self.agent_view_size))
        if self.detect_traces:
            for trace_pos in self.agent_trace_sets[agent_id]:
                self.target_each_map[agent_id][trace_pos[1]+self.agent_view_size, trace_pos[0]+self.agent_view_size] = 0.25
                self.global_target_each_map[agent_id][trace_pos[1]+self.agent_view_size,trace_pos[0]+self.agent_view_size] = 0.25
        for target_pos in self.agent_target_sets[agent_id]:
            self.target_each_map[agent_id][target_pos[1]+self.agent_view_size, target_pos[0]+self.agent_view_size] = 1
            # self.global_target_each_map[agent_id][target_pos[1]+self.agent_view_size-1:target_pos[1]+self.agent_view_size+2,
            #                         target_pos[0]+self.agent_view_size-1:target_pos[0]+self.agent_view_size+2] = 1
            self.global_target_each_map[agent_id][target_pos[1]+self.agent_view_size,
                                    target_pos[0]+self.agent_view_size] = 1
        

    def set_target_all_map(self):
        self.target_all_map = np.zeros((self.width + 2*self.agent_view_size, self.height + 2*self.agent_view_size))
        self.global_target_all_map = np.zeros((self.width + 2*self.agent_view_size, self.height + 2*self.agent_view_size))
        if self.detect_traces:
            for trace_pos in self.all_trace_set:
                self.target_all_map[trace_pos[1]+self.agent_view_size, trace_pos[0]+self.agent_view_size] = 0.25
                self.global_target_all_map[trace_pos[1]+self.agent_view_size,trace_pos[0]+self.agent_view_size] = 0.25
    
        for target_pos in self.all_target_set:
            self.target_all_map[target_pos[1]+self.agent_view_size, target_pos[0]+self.agent_view_size] = 1
            # self.global_target_all_map[target_pos[1]+self.agent_view_size-1:target_pos[1]+self.agent_view_size+2,
            #                         target_pos[0]+self.agent_view_size-1:target_pos[0]+self.agent_view_size+2] = 1
            self.global_target_all_map[target_pos[1]+self.agent_view_size,
                                    target_pos[0]+self.agent_view_size] = 1
        

    def set_rubble_each_map(self, agent_id):

        self.rubble_each_map[agent_id] = np.zeros((self.width + 2*self.agent_view_size, self.height + 2*self.agent_view_size))
        self.global_rubble_each_map[agent_id] = np.zeros((self.width + 2*self.agent_view_size, self.height + 2*self.agent_view_size))
        for rubble_pos in self.agent_rubble_sets[agent_id]:
            self.rubble_each_map[agent_id][rubble_pos[1]+self.agent_view_size, rubble_pos[0]+self.agent_view_size] = 1 #1/2
            self.global_rubble_each_map[agent_id][rubble_pos[1]+self.agent_view_size-1:rubble_pos[1]+self.agent_view_size+2,
                                    rubble_pos[0]+self.agent_view_size-1:rubble_pos[0]+self.agent_view_size+2] = 1 #1/2
            
    def set_rubble_all_map(self): 
        self.rubble_all_map = np.zeros((self.width + 2*self.agent_view_size, self.height + 2*self.agent_view_size))
        self.global_rubble_all_map = np.zeros((self.width + 2*self.agent_view_size, self.height + 2*self.agent_view_size))
        for rubble_pos in self.all_rubble_set:
            self.rubble_all_map[rubble_pos[1]+self.agent_view_size, rubble_pos[0]+self.agent_view_size] = 1 #1/2
            self.global_rubble_all_map[rubble_pos[1]+self.agent_view_size-1:rubble_pos[1]+self.agent_view_size+2,
                                    rubble_pos[0]+self.agent_view_size-1:rubble_pos[0]+self.agent_view_size+2] = 1 #1/2
            

    def reset(self):
        #TODO: make the grids smaller because 360 vision doesn't require 2*agent_view_size padding
        self.explorable_size = 0
        # Agents
        self.agent_groups = np.eye(self.num_agents)
        self.agent_types_list = np.array(self.original_agent_types_list)
        for index in range(self.num_agents):
            if self.original_agent_types_list[index] == 10: # replace the code 10 agent randomly with 0 or 1 or 2
                new_type = self._rand_int(0, 2)
                self.agent_types_list[index] = new_type
        self.agent_count = np.zeros((self.n_agent_types)) # count of each agent type
        self.agent_alive = np.ones((self.num_agents))
        
        obs = MiniGridEnv.reset(self, choose=True)

        if self.target_pos[0] == -1: # or self.initial_trace[0] == -1:
            self.agent_alive = np.zeros((self.num_agents))
            print("Target can not be reached, ignoring this map.")
        
        self.target_pos = None # Assuming there is a single target
        self.current_target_pos = None
        self.num_step = 0
        self.episode_no += 1

        # init local map
        self.explored_each_map = []
        self.obstacle_each_map = []
        self.previous_explored_each_map = []
        current_agent_pos = []
        self.target_each_map = []
        self.global_target_each_map = []
        self.rubble_each_map = []
        self.global_rubble_each_map = []
        self.each_agent_trajectory_map = []
        agent_local_views = []
        
        self.target_found = np.zeros((self.num_agents))
        self.is_target_found = 0
        self.target_found_step = np.nan
        self.found_switch = 0
        self.target_rescued = 0
        self.target_rescued_step = np.nan
        self.stayed_switch = 0
        
        
        
        agent_rubble_sets = [] # temporary set
        self.agent_rubble_sets = [] # permanent set
        self.all_rubble_set = set()
        
        self.agent_trace_sets = []
        self.all_trace_set = set()

        agent_target_sets = [] # temporary set
        self.agent_target_sets = []
        self.all_target_set = set()
       
        # self.paths = [[] for _ in range(self.num_agents)]
        
        # APF repeat penalty.
        self.ft_goals = [None for _ in range(self.num_agents)]
        self.apf_penalty = np.zeros((
            self.num_agents,
            self.width + 2*self.agent_view_size,
            self.height + 2*self.agent_view_size
        ))

        for i in range(self.num_agents):

            self.agent_count[self.agent_types_list[i]] += 1
           
            agent_rubble_sets.append(set()) 
            agent_target_sets.append(set())
            self.agent_rubble_sets.append(set())
            self.agent_trace_sets.append(set())
            self.agent_target_sets.append(set())

            self.explored_each_map.append(
                np.zeros((self.width + 2*self.agent_view_size, self.height + 2*self.agent_view_size)))
            self.obstacle_each_map.append(
                np.zeros((self.width + 2*self.agent_view_size, self.height + 2*self.agent_view_size)))
            self.rubble_each_map.append(
                np.zeros((self.width + 2*self.agent_view_size, self.height + 2*self.agent_view_size)))
            self.global_rubble_each_map.append(
                np.zeros((self.width + 2*self.agent_view_size, self.height + 2*self.agent_view_size)))
            self.previous_explored_each_map.append(
                np.zeros((self.width + 2*self.agent_view_size, self.height + 2*self.agent_view_size)))
            self.target_each_map.append(
                np.zeros((self.width + 2*self.agent_view_size, self.height + 2*self.agent_view_size)))
            self.global_target_each_map.append(
                np.zeros((self.width + 2*self.agent_view_size, self.height + 2*self.agent_view_size)))
            self.each_agent_trajectory_map.append(
                np.zeros((self.width + 2*self.agent_view_size, self.height + 2*self.agent_view_size)))

            

            pos = [self.agent_pos[i][1] + self.agent_view_size,
                   self.agent_pos[i][0] + self.agent_view_size]
            current_agent_pos.append(pos)
            direction = self.agent_dir[i]
            target_pos = None

            ### Front camera view
            # local_map = np.rot90(obs[i]['image'][:, :, 0].T, 3)
            # local_map = np.rot90(local_map, 4-direction) # adjust angle

            ### 360 degrees view
            local_map = obs[i]['image'][:, :, 0].T
            for j in range(self.num_agents): #adding seen agents to the local view
                if j != i:
                    relative_pos = [self.agent_pos[j][0] - self.agent_pos[i][0], self.agent_pos[j][1] - self.agent_pos[i][1]]
                    if abs(relative_pos[0]) <= self.agent_view_size//2 and abs(relative_pos[1]) <= self.agent_view_size//2:
                        agent_cell = [relative_pos[1] + self.agent_view_size//2, relative_pos[0] + self.agent_view_size//2]
                        if local_map[agent_cell[0], agent_cell[1]] != 0:
                            local_map[agent_cell[0], agent_cell[1]] = 220
            agent_local_views.append(local_map)

            # if self.agent_types_list[i] == 0: # actuator agents have a smaller view size
            #     agent_view_size = self.agent_view_size
            # else:
            agent_view_size = self.agent_view_size

            # if not self.agent_alive[i]: # this should not be trigerred, every agent is alive at the first step
            #     continue

            ### Front camera view
            # for x in range(agent_view_size):
            #         for y in range(agent_view_size):
            #             if local_map[x][y] == 0:
            #                 continue
            #             elif direction == 0: # Facing right
            #                 self.explored_each_map[i][x+pos[0] -
            #                                         agent_view_size//2][y+pos[1]] = 1
            #                 if local_map[x][y] != 20:
            #                     if local_map[x][y] == 180:
            #                         target_pos = [y+pos[1]-agent_view_size, x+pos[0] - 3*agent_view_size//2]
            #                         self.target_pos = target_pos
            #                         self.current_target_pos = [x+pos[0] -
            #                                                 agent_view_size//2 , y+pos[1]]
            #                         agent_target_sets[i].add(tuple([y+pos[1]-agent_view_size, x+pos[0] - 3*agent_view_size//2]))
            #                     elif local_map[x][y] == 60:
            #                         self.agent_trace_sets[i].add(tuple([y+pos[1]-agent_view_size, x+pos[0] - 3*agent_view_size//2]))
            #                     elif local_map[x][y] == 240:
            #                         agent_rubble_sets[i].add(tuple([y+pos[1]-agent_view_size, x+pos[0] - 3*agent_view_size//2]))
            #                     elif local_map[x][y] == 80:
            #                         pass
            #                     else:
            #                         self.obstacle_each_map[i][x+pos[0] - agent_view_size//2][y+pos[1]] = 1
            #             elif direction == 1: # Facing down
            #                 self.explored_each_map[i][x+pos[0]][y+pos[1] -
            #                                                     agent_view_size//2] = 1
            #                 if local_map[x][y] != 20:
            #                     if local_map[x][y] == 180:
            #                         target_pos = [y+pos[1]-3*agent_view_size//2, x+pos[0] - agent_view_size]
            #                         self.target_pos = target_pos
            #                         self.current_target_pos = [x+pos[0], y+pos[1] -
            #                                                 agent_view_size//2]
            #                         agent_target_sets[i].add(tuple([y+pos[1]-3*agent_view_size//2, x+pos[0] - agent_view_size]))
            #                     elif local_map[x][y] == 60:
            #                         self.agent_trace_sets[i].add(tuple([y+pos[1]-3*agent_view_size//2, x+pos[0] - agent_view_size]))
            #                     elif local_map[x][y] == 240:
            #                         agent_rubble_sets[i].add(tuple([y+pos[1]-3*agent_view_size//2, x+pos[0] - agent_view_size]))
            #                     elif local_map[x][y] == 80:
            #                         pass
            #                     else:
            #                         self.obstacle_each_map[i][x+pos[0]][y+pos[1] - agent_view_size//2] = 1
            #             elif direction == 2: # Facing left
            #                 self.explored_each_map[i][x+pos[0]-agent_view_size //
            #                                         2][y+pos[1]-agent_view_size+1] = 1
            #                 if local_map[x][y] != 20:
            #                     if local_map[x][y] == 180:
            #                         target_pos = [y+pos[1]-2*agent_view_size+1, x+pos[0] - 3*agent_view_size//2]
            #                         self.target_pos = target_pos
            #                         self.current_target_pos = [x+pos[0]-agent_view_size//2, y+pos[1] -
            #                                                 agent_view_size+1]
            #                         agent_target_sets[i].add(tuple([y+pos[1]-2*agent_view_size+1, x+pos[0] - 3*agent_view_size//2]))
            #                     elif local_map[x][y] == 60:
            #                         self.agent_trace_sets[i].add(tuple([y+pos[1]-2*agent_view_size+1, x+pos[0] - 3*agent_view_size//2]))
            #                     elif local_map[x][y] == 240:
            #                         agent_rubble_sets[i].add(tuple([y+pos[1]-2*agent_view_size+1, x+pos[0] - 3*agent_view_size//2]))
            #                     elif local_map[x][y] == 80:
            #                         pass
            #                     else:
            #                         self.obstacle_each_map[i][x+pos[0]-agent_view_size // 2][y+pos[1]-agent_view_size+1] = 1
            #             elif direction == 3: # Facing up 
            #                 self.explored_each_map[i][x+pos[0]-agent_view_size +
            #                                         1][y+pos[1]-agent_view_size//2] = 1
            #                 if local_map[x][y] != 20:
            #                     if local_map[x][y] == 180:
            #                         target_pos = [y+pos[1]-3*agent_view_size//2, x+pos[0] - 2*agent_view_size+1]
            #                         self.target_pos = target_pos
            #                         self.current_target_pos = [x+pos[0]-agent_view_size + 1, y+pos[1] -
            #                                                    agent_view_size//2]
            #                         agent_target_sets[i].add(tuple([y+pos[1]-3*agent_view_size//2, x+pos[0] - 2*agent_view_size+1]))
            #                     elif local_map[x][y] == 60:
            #                         self.agent_trace_sets[i].add(tuple([y+pos[1]-3*agent_view_size//2, x+pos[0] - 2*agent_view_size+1]))
            #                     elif local_map[x][y] == 240:
            #                         agent_rubble_sets[i].add(tuple([y+pos[1]-3*agent_view_size//2, x+pos[0] - 2*agent_view_size+1]))
            #                     elif local_map[x][y] == 80:
            #                         pass
            #                     else:
            #                         self.obstacle_each_map[i][x+pos[0]-agent_view_size + 1][y+pos[1]-agent_view_size//2] = 1

            ### 360 degrees view
            for x in range(agent_view_size):
                for y in range(agent_view_size):
                    if local_map[x][y] == 0:
                        continue
                    else:
                        self.explored_each_map[i][x+pos[0] - agent_view_size//2][y+pos[1] - agent_view_size//2] = 1
                        if local_map[x][y] != 20:
                            if local_map[x][y] == 180:
                                target_pos = [y+pos[1]-3*agent_view_size//2, x+pos[0] - 3*agent_view_size//2]
                                # self.target_pos = target_pos
                                self.current_target_pos = [x+pos[0] -
                                                        agent_view_size//2 , y+pos[1] - agent_view_size//2]
                                agent_target_sets[i].add(tuple([y+pos[1]-3*agent_view_size//2, x+pos[0] - 3*agent_view_size//2]))
                            elif local_map[x][y] == 60:
                                self.agent_trace_sets[i].add(tuple([y+pos[1]-3*agent_view_size//2, x+pos[0] - 3*agent_view_size//2]))
                            elif local_map[x][y] == 240:
                                agent_rubble_sets[i].add(tuple([y+pos[1]-3*agent_view_size//2, x+pos[0] - 3*agent_view_size//2]))
                            elif local_map[x][y] == 80: #door
                                pass
                            elif local_map[x][y] == 220: #another agent
                                pass
                            else:
                                self.obstacle_each_map[i][x+pos[0] - agent_view_size//2][y+pos[1] - agent_view_size//2] = 1
                        
            
            # Agent i has found the target
            if target_pos != None:
                self.target_found[i] = 1
                self.is_target_found = 1
                
            if self.use_partial_comm:
                # Forming the agent groups
                for j in range(i+1, self.num_agents):
                    # if self.agent_alive[j]:
                    d_ij = euclideandistance(self.agent_pos[i], self.agent_pos[j])
                    comm_prob = np.exp(-d_ij**2/(self.com_sigma**2))
                    # Sample from this probability
                    if self._rand_float(0, 1) < comm_prob:
                        self.agent_groups[i,j] = 1
                        self.agent_groups[j,i] = 1
        
        explored_all_map = np.zeros((self.width + 2*self.agent_view_size,
                                    self.height + 2*self.agent_view_size))
        obstacle_all_map = np.zeros((self.width + 2*self.agent_view_size,
                                    self.height + 2*self.agent_view_size))

        # self.previous_all_map = np.zeros(
        #     (self.width + 2*self.agent_view_size, self.width + 2*self.agent_view_size))


        for i in range(self.num_agents):
            self.each_agent_trajectory_map[i][current_agent_pos[i][0]-1:current_agent_pos[i][0]+2, 
                                        current_agent_pos[i][1]-1:current_agent_pos[i][1]+2] = 1
            # self.each_agent_trajectory_map[i][current_agent_pos[i][0], current_agent_pos[i][1]] = 1
            explored_all_map = np.maximum( explored_all_map, self.explored_each_map[i])
            obstacle_all_map = np.maximum( obstacle_all_map, self.obstacle_each_map[i])
        
        
     
        if self.use_partial_comm:
            connected_agent_groups = get_connected_agents(self.agent_groups) # connected agents encompass agent groups
            self.explored_shared_map = [] # the number of shared maps is equal to the number of connected agent groups
            self.obstacle_shared_map = []
            self.target_found_shared = []
            shared_rubble_sets = []
            shared_trace_sets = []
            shared_target_sets = []

            counter = 0
            for group in connected_agent_groups:
                explored_shared_map = np.zeros((self.width + 2*self.agent_view_size, self.height + 2*self.agent_view_size))
                obstacle_shared_map = np.zeros((self.width + 2*self.agent_view_size, self.height + 2*self.agent_view_size))
                target_found_shared = 0
                shared_rubble_sets.append(set())
                shared_trace_sets.append(set())
                shared_target_sets.append(set())
                for i in group:
                    explored_shared_map = np.maximum(explored_shared_map, self.explored_each_map[i])
                    obstacle_shared_map = np.maximum(obstacle_shared_map, self.obstacle_each_map[i])
                    target_found_shared = np.maximum(target_found_shared, self.target_found[i])
                    shared_rubble_sets[counter] = shared_rubble_sets[counter].union(agent_rubble_sets[i])
                    shared_trace_sets[counter] = shared_trace_sets[counter].union(self.agent_trace_sets[i])
                    shared_target_sets[counter] = shared_target_sets[counter].union(agent_target_sets[i])
                self.explored_shared_map.append(explored_shared_map)
                self.obstacle_shared_map.append(obstacle_shared_map)
                self.target_found_shared.append(target_found_shared)
                counter += 1
        elif self.use_full_comm:
            connected_agent_groups = [[agent_id for agent_id in range(self.num_agents)]]
        
    
        # based on agent communications, each_maps and other info are shared and target_each_maps are formed.
        if self.use_full_comm:
            for i in range(self.num_agents):
                if self.target_found.any():
                    self.target_found[i] = 1
                
                self.all_trace_set = self.all_trace_set.union(self.agent_trace_sets[i])
                self.all_rubble_set = self.all_rubble_set.union(agent_rubble_sets[i])
                self.all_target_set = self.all_target_set.union(agent_target_sets[i])
                
                self.explored_each_map[i] = explored_all_map.copy()
                self.obstacle_each_map[i] = obstacle_all_map.copy()
                self.previous_explored_each_map[i] = self.explored_each_map[i] 
            for i in range(self.num_agents):
                self.agent_trace_sets[i] = self.all_trace_set
                self.agent_target_sets[i] = self.all_target_set
                # self.set_target_each_map(i)
                self.agent_rubble_sets[i] = self.all_rubble_set
                # self.set_rubble_each_map(i)

        elif self.use_partial_comm:
            counter = 0
            for group in connected_agent_groups:
                for i in group:
                    # for j in group:
                    #     if self.agent_rubble_sets[i] != None and self.rubble_cells_attended[j] != None: 
                    #         self.agent_rubble_sets[i].difference_update(self.rubble_cells_attended[j])

                    self.target_found[i] = self.target_found_shared[counter]
                    self.agent_trace_sets[i] = self.agent_trace_sets[i].union(shared_trace_sets[counter])
                    self.agent_target_sets[i] = self.agent_target_sets[i].union(shared_target_sets[counter])
                    self.set_target_each_map(i)

                    self.agent_rubble_sets[i] = self.agent_rubble_sets[i].union(shared_rubble_sets[counter])
                    self.all_rubble_set = self.all_rubble_set.union(self.agent_rubble_sets[i])
                    self.set_rubble_each_map(i)

                    self.explored_each_map[i] = self.explored_shared_map[counter]
                    self.obstacle_each_map[i] = self.obstacle_shared_map[counter]
                    self.previous_explored_each_map[i] = self.explored_each_map[i] 

                counter += 1
        self.set_target_all_map()
        self.set_rubble_all_map()
        
        self.explored_map = np.array(explored_all_map).astype(int)[
            self.agent_view_size: self.width+self.agent_view_size, self.agent_view_size: self.width+self.agent_view_size]
        
        
        occupied_each_map = np.copy(self.obstacle_each_map)
        for i in range(self.num_agents):
            for rubble_pos in self.agent_rubble_sets[i]:
                occupied_each_map[i][rubble_pos[1]+self.agent_view_size, rubble_pos[0]+self.agent_view_size] = 1
            if self.target_found[i]:
                occupied_each_map[i][self.target_pos[1]+self.agent_view_size, self.target_pos[0]+self.agent_view_size] = 1
        occupied_all_map = np.copy(obstacle_all_map)
        for rubble_pos in self.all_rubble_set:
            occupied_all_map[rubble_pos[1]+self.agent_view_size, rubble_pos[0]+self.agent_view_size] = 1
        if self.is_target_found:
            occupied_all_map[self.target_pos[1]+self.agent_view_size, self.target_pos[0]+self.agent_view_size] = 1
        
        number_of_rubbles_removed = 0
        agent_number_of_known_rubbles = np.zeros(self.num_agents)
        for i in range(self.num_agents):
            agent_number_of_known_rubbles[i] = len(self.agent_rubble_sets[i])
        
        # APF penalty
        for i in range(self.num_agents):
            x, y = current_agent_pos[i]
            self.apf_penalty[i, x, y] = 5.0  # constant

        self.info = {}
        self.info['explored_all_map'] = np.array(explored_all_map)
        self.info['agent_pos'] = np.array(self.agent_pos)
        self.info['current_agent_pos'] = np.array(current_agent_pos)
        self.info['agent_direction'] = np.array(self.agent_dir)
        self.info['explored_each_map'] = np.array(self.explored_each_map)
        self.info['occupied_all_map'] = np.array(occupied_all_map)
        self.info['occupied_each_map'] = np.array(occupied_each_map)
        self.info['rubble_each_map'] = np.array(self.rubble_each_map)
        self.info['global_rubble_each_map'] = np.array(self.global_rubble_each_map)
        self.info['rubble_all_map'] = np.array(self.rubble_all_map)
        self.info['global_rubble_all_map'] = np.array(self.global_rubble_all_map)
        self.info['target_each_map'] = np.array(self.target_each_map)
        self.info['global_target_each_map'] = np.array(self.global_target_each_map)
        self.info['target_all_map'] = np.array(self.target_all_map)
        self.info['global_target_all_map'] = np.array(self.global_target_all_map)
        self.info['each_agent_trajectory_map'] = self.each_agent_trajectory_map
        # self.info['main_agent_trajectory_map'] = self.main_agent_trajectory_map
        # self.info['helper_agents_trajectory_map'] = self.helper_agents_trajectory_map

        self.info['merge_explored_ratio'] = self.merge_ratio
        self.info['agent_explored_reward'] = self.agent_exploration_reward
        self.info['target_rescued'] = self.target_rescued
        self.info['target_rescued_step'] = self.target_rescued_step
        self.info['target_found'] = self.target_found
        self.info['is_target_found'] = self.is_target_found
        self.info['target_found_step'] = self.target_found_step
        # self.info['agent_paths'] = self.paths
        self.info['agent_groups'] = self.agent_groups
        self.info['connected_agent_groups'] = connected_agent_groups
        self.info['agent_alive'] = self.agent_alive
        self.info['agent_types_list'] = self.agent_types_list
        # self.info['agent_inventory'] = self.agent_inventory
        self.info['number_of_rubbles_removed'] = number_of_rubbles_removed
        self.info['agent_number_of_known_rubbles'] = agent_number_of_known_rubbles
        self.info['agent_local_views'] = agent_local_views
        self.info['agent_target_sets'] = self.agent_target_sets

        # self.info['agent_rubble_sets'] = self.agent_rubble_sets
        # self.info['agent_target_sets'] = self.agent_target_sets
        # self.info['agent_trace_sets'] = self.agent_trace_sets
        # self.info['all_rubble_set'] = self.all_rubble_set
        # self.info['all_target_set'] = self.all_target_set
        # self.info['all_trace_set'] = self.all_trace_set

        
        self.merge_ratio = 0
        # self.merge_reward = 0
        self.agent_exploration_reward = np.zeros((self.num_agents))
        return obs, self.info

    def step(self, action):
        obs, reward, done = MiniGridEnv.step(self, action)
        self.explored_each_map_t = []
        self.obstacle_each_map_t = []
        self.target_each_map_t = []
        current_agent_pos = []
        self.num_step += 1

        reward_explored_each_map = np.zeros(
            (self.num_agents, self.width + 2*self.agent_view_size, self.height + 2*self.agent_view_size))
        step_reward_each_map = np.zeros(
            (self.num_agents, self.width + 2*self.agent_view_size, self.height + 2*self.agent_view_size))
        explored_all_map = np.zeros((self.width + 2*self.agent_view_size,
                                    self.height + 2*self.agent_view_size))
        obstacle_all_map = np.zeros((self.width + 2*self.agent_view_size,
                                    self.height + 2*self.agent_view_size))

        self.agent_groups = np.eye(self.num_agents)
        agent_rubble_sets = [] # temporary set
        agent_target_sets = [] # temporary set
        agent_local_views = []

            
        for i in range(self.num_agents):

            agent_rubble_sets.append(set()) 
            agent_target_sets.append(set())
            self.explored_each_map_t.append(
                np.zeros((self.width + 2*self.agent_view_size, self.height + 2*self.agent_view_size)))
            self.obstacle_each_map_t.append(
                np.zeros((self.width + 2*self.agent_view_size, self.height + 2*self.agent_view_size)))
            
            
            
            pos = [self.agent_pos[i][1] + self.agent_view_size,
                   self.agent_pos[i][0] + self.agent_view_size]
            current_agent_pos.append(pos)
            direction = self.agent_dir[i]
            target_pos = None

            ### Front camera view
            # local_map = np.rot90(obs[i]['image'][:, :, 0].T, 3)
            # local_map = np.rot90(local_map, 4-direction) # adjust angle

            ### 360 degrees view
            local_map = obs[i]['image'][:, :, 0].T
            for j in range(self.num_agents): #adding seen agents to the local view
                if j != i:
                    relative_pos = [self.agent_pos[j][0] - self.agent_pos[i][0], self.agent_pos[j][1] - self.agent_pos[i][1]]
                    if abs(relative_pos[0]) <= self.agent_view_size//2 and abs(relative_pos[1]) <= self.agent_view_size//2:
                        agent_cell = [relative_pos[1] + self.agent_view_size//2, relative_pos[0] + self.agent_view_size//2]
                        if local_map[agent_cell[0], agent_cell[1]] != 0:
                            local_map[agent_cell[0], agent_cell[1]] = 220
            agent_local_views.append(local_map)



            # if self.agent_types_list[i] == 0: # actuator agents have a smaller view size
            #     agent_view_size = self.agent_view_size
            #     # agent_view_size = self.agent_view_size // 2
            # else:
            agent_view_size = self.agent_view_size

            if not self.agent_alive[i]:
                continue

            ### Front camera view
            # for x in range(agent_view_size):
            #         for y in range(agent_view_size):
            #             if local_map[x][y] == 0:
            #                 continue
            #             elif direction == 0: # Facing right
            #                 self.explored_each_map_t[i][x+pos[0] -
            #                                         agent_view_size//2][y+pos[1]] = 1
            #                 if local_map[x][y] != 20:
            #                     if local_map[x][y] == 180:
            #                         target_pos = [y+pos[1]-agent_view_size, x+pos[0] - 3*agent_view_size//2]
            #                         self.target_pos = target_pos
            #                         self.current_target_pos = [x+pos[0] -
            #                                                 agent_view_size//2 , y+pos[1]]
            #                         agent_target_sets[i].add(tuple([y+pos[1]-agent_view_size, x+pos[0] - 3*agent_view_size//2]))
            #                     elif local_map[x][y] == 60:
            #                         self.agent_trace_sets[i].add(tuple([y+pos[1]-agent_view_size, x+pos[0] - 3*agent_view_size//2]))
            #                     elif local_map[x][y] == 240:
            #                         agent_rubble_sets[i].add(tuple([y+pos[1]-agent_view_size, x+pos[0] - 3*agent_view_size//2]))
            #                     elif local_map[x][y] == 80:
            #                         pass
            #                     else:
            #                         self.obstacle_each_map_t[i][x+pos[0] - agent_view_size//2][y+pos[1]] = 1
            #                 else:
            #                     if tuple([y+pos[1]-agent_view_size, x+pos[0] - 3*agent_view_size//2]) in self.agent_rubble_sets[i]:
            #                         self.agent_rubble_sets[i].discard(tuple([y+pos[1]-agent_view_size, x+pos[0] - 3*agent_view_size//2]))
            #             elif direction == 1: # Facing down
            #                 self.explored_each_map_t[i][x+pos[0]][y+pos[1] -
            #                                                     agent_view_size//2] = 1
            #                 if local_map[x][y] != 20:
            #                     if local_map[x][y] == 180:
            #                         target_pos = [y+pos[1]-3*agent_view_size//2, x+pos[0] - agent_view_size]
            #                         self.target_pos = target_pos
            #                         self.current_target_pos = [x+pos[0], y+pos[1] -
            #                                                 agent_view_size//2]
            #                         agent_target_sets[i].add(tuple([y+pos[1]-3*agent_view_size//2, x+pos[0] - agent_view_size]))
            #                     elif local_map[x][y] == 60:
            #                         self.agent_trace_sets[i].add(tuple([y+pos[1]-3*agent_view_size//2, x+pos[0] - agent_view_size]))
            #                     elif local_map[x][y] == 240:
            #                         agent_rubble_sets[i].add(tuple([y+pos[1]-3*agent_view_size//2, x+pos[0] - agent_view_size]))
            #                     elif local_map[x][y] == 80:
            #                         pass
            #                     else:
            #                         self.obstacle_each_map_t[i][x+pos[0]][y+pos[1] - agent_view_size//2] = 1
            #                 else:
            #                     if tuple([y+pos[1]-3*agent_view_size//2, x+pos[0] - agent_view_size]) in self.agent_rubble_sets[i]:
            #                         self.agent_rubble_sets[i].discard(tuple([y+pos[1]-3*agent_view_size//2, x+pos[0] - agent_view_size]))
            #             elif direction == 2: # Facing left
            #                 self.explored_each_map_t[i][x+pos[0]-agent_view_size //
            #                                         2][y+pos[1]-agent_view_size+1] = 1
            #                 if local_map[x][y] != 20:
            #                     if local_map[x][y] == 180:
            #                         target_pos = [y+pos[1]-2*agent_view_size+1, x+pos[0] - 3*agent_view_size//2]
            #                         self.target_pos = target_pos
            #                         self.current_target_pos = [x+pos[0]-agent_view_size//2, y+pos[1] -
            #                                                 agent_view_size+1]
            #                         agent_target_sets[i].add(tuple([y+pos[1]-2*agent_view_size+1, x+pos[0] - 3*agent_view_size//2]))
            #                     elif local_map[x][y] == 60:
            #                         self.agent_trace_sets[i].add(tuple([y+pos[1]-2*agent_view_size+1, x+pos[0] - 3*agent_view_size//2]))
            #                     elif local_map[x][y] == 240:
            #                         agent_rubble_sets[i].add(tuple([y+pos[1]-2*agent_view_size+1, x+pos[0] - 3*agent_view_size//2]))
            #                     elif local_map[x][y] == 80:
            #                         pass
            #                     else:
            #                         self.obstacle_each_map_t[i][x+pos[0]-agent_view_size // 2][y+pos[1]-agent_view_size+1] = 1
            #                 else:
            #                     if tuple([y+pos[1]-2*agent_view_size+1, x+pos[0] - 3*agent_view_size//2]) in self.agent_rubble_sets[i]:
            #                         self.agent_rubble_sets[i].discard(tuple([y+pos[1]-2*agent_view_size+1, x+pos[0] - 3*agent_view_size//2]))
            #             elif direction == 3: # Facing up 
            #                 self.explored_each_map_t[i][x+pos[0]-agent_view_size +
            #                                         1][y+pos[1]-agent_view_size//2] = 1
            #                 if local_map[x][y] != 20:
            #                     if local_map[x][y] == 180:
            #                         target_pos = [y+pos[1]-3*agent_view_size//2, x+pos[0] - 2*agent_view_size+1]
            #                         self.target_pos = target_pos
            #                         self.current_target_pos = [x+pos[0]-agent_view_size + 1, y+pos[1] -
            #                                                 agent_view_size//2]
            #                         agent_target_sets[i].add(tuple([y+pos[1]-3*agent_view_size//2, x+pos[0] - 2*agent_view_size+1]))
            #                     elif local_map[x][y] == 60:
            #                         self.agent_trace_sets[i].add(tuple([y+pos[1]-3*agent_view_size//2, x+pos[0] - 2*agent_view_size+1]))
            #                     elif local_map[x][y] == 240:
            #                         agent_rubble_sets[i].add(tuple([y+pos[1]-3*agent_view_size//2, x+pos[0] - 2*agent_view_size+1]))
            #                     elif local_map[x][y] == 80:
            #                         pass
            #                     else:
            #                         self.obstacle_each_map_t[i][x+pos[0]-agent_view_size + 1][y+pos[1]-agent_view_size//2] = 1
            #                 else:
            #                     if tuple([y+pos[1]-3*agent_view_size//2, x+pos[0] - 2*agent_view_size+1]) in self.agent_rubble_sets[i]:
            #                         self.agent_rubble_sets[i].discard(tuple([y+pos[1]-3*agent_view_size//2, x+pos[0] - 2*agent_view_size+1]))

            ### 360 degrees view
            for x in range(agent_view_size):
                for y in range(agent_view_size):
                    if local_map[x][y] == 0:
                        continue
                    else:
                        self.explored_each_map[i][x+pos[0] - agent_view_size//2][y+pos[1] - agent_view_size//2] = 1
                        if local_map[x][y] != 20:
                            if local_map[x][y] == 180:
                                target_pos = [y+pos[1]-3*agent_view_size//2, x+pos[0] - 3*agent_view_size//2]
                                self.target_pos = target_pos
                                self.current_target_pos = [x+pos[0] - agent_view_size//2 , y+pos[1] - agent_view_size//2]
                                agent_target_sets[i].add(tuple([y+pos[1]-3*agent_view_size//2, x+pos[0] - 3*agent_view_size//2]))
                            elif local_map[x][y] == 60:
                                self.agent_trace_sets[i].add(tuple([y+pos[1]-3*agent_view_size//2, x+pos[0] - 3*agent_view_size//2]))
                            elif local_map[x][y] == 240:
                                agent_rubble_sets[i].add(tuple([y+pos[1]-3*agent_view_size//2, x+pos[0] - 3*agent_view_size//2]))
                            elif local_map[x][y] == 80: #door
                                pass
                            elif local_map[x][y] == 220: #another agent
                                pass
                            else:
                                self.obstacle_each_map[i][x+pos[0] - agent_view_size//2][y+pos[1] - agent_view_size//2] = 1
                        elif tuple([y+pos[1]-3*agent_view_size//2, x+pos[0] - 3*agent_view_size//2]) in self.agent_rubble_sets[i]:
                                self.agent_rubble_sets[i].discard(tuple([y+pos[1]-3*agent_view_size//2, x+pos[0] - 3*agent_view_size//2]))
            
            # Agent i has found the target
            if target_pos != None:
                self.target_found[i] = 1
                if self.is_target_found == 0:
                    self.is_target_found = 1
                    alpha = 1/self.num_agents
                    reward += 100*self._reward() # Team reward for finding the target
                    # reward[i] += 50*self._reward() # Individual reward for finding the target
                    # reward += alpha*100*self._reward() # Team reward for finding the target
                    # reward[i] += (1-alpha)*100*self._reward() # Individual reward for finding the target

            if self.use_partial_comm:
                # Forming the agent groups
                for j in range(i+1, self.num_agents):
                    if self.agent_alive[j]:
                        d_ij = euclideandistance(self.agent_pos[i], self.agent_pos[j])
                        comm_prob = np.exp(-d_ij**2/(self.com_sigma**2))
                        # Sample from this probability
                        if self._rand_float(0, 1) < comm_prob:
                            self.agent_groups[i,j] = 1
                            self.agent_groups[j,i] = 1
                                    
        
        for i in range(self.num_agents):
            if not self.agent_alive[i]:
                continue
            # Update the explored and obstacle each maps
            self.explored_each_map[i] = np.maximum(
                self.explored_each_map[i], self.explored_each_map_t[i])
            self.obstacle_each_map[i] = np.maximum(
                self.obstacle_each_map[i], self.obstacle_each_map_t[i])

            
            # Update the explored and obstacle all maps
            explored_all_map = np.maximum(explored_all_map, self.explored_each_map[i])
            obstacle_all_map = np.maximum(obstacle_all_map, self.obstacle_each_map[i])

            # Reward Calculation
            reward_explored_each_map[i] = self.explored_each_map[i].copy()
            reward_explored_each_map[i][reward_explored_each_map[i] != 0] = 1

            reward_previous_explored_each_map = self.previous_explored_each_map[i].copy()
            reward_previous_explored_each_map[reward_previous_explored_each_map != 0] = 1

            # reward_obstacle_each_map[i] = self.obstacle_each_map[i].copy()
            # reward_obstacle_each_map[i][reward_obstacle_each_map[i] != 0] = 1

            # delta_reward_each_map[i] = reward_explored_each_map[i] - reward_obstacle_each_map[i]
            # step_reward_each_map[i] = np.array(delta_reward_each_map[i]) - np.array(reward_previous_explored_each_map)
            step_reward_each_map[i] = np.array(reward_explored_each_map[i]) - np.array(reward_previous_explored_each_map)
            

            # forming the trajectory maps
            self.each_agent_trajectory_map[i] = np.round(self.each_agent_trajectory_map[i]*self.trajectory_forget_rate, 4)
            self.each_agent_trajectory_map[i][current_agent_pos[i][0]-1:current_agent_pos[i][0]+2, 
                                        current_agent_pos[i][1]-1:current_agent_pos[i][1]+2] = 1
            # self.each_agent_trajectory_map[i][current_agent_pos[i][0], current_agent_pos[i][1]] = 1
        
        
        if self.use_partial_comm:
            connected_agent_groups = get_connected_agents(self.agent_groups)
            self.explored_shared_map = [] # the number of shared maps is equal to the number of connected agent groups
            self.obstacle_shared_map = []
            self.target_found_shared = []
            shared_rubble_sets = []
            shared_trace_sets = []
            shared_target_sets = []
            counter = 0
            for group in connected_agent_groups:
                explored_shared_map = np.zeros((self.width + 2*self.agent_view_size, self.height + 2*self.agent_view_size))
                obstacle_shared_map = np.zeros((self.width + 2*self.agent_view_size, self.height + 2*self.agent_view_size))
                target_found_shared = 0
                shared_rubble_sets.append(set())
                shared_trace_sets.append(set())
                shared_target_sets.append(set())
                for i in group:
                    explored_shared_map = np.maximum(explored_shared_map, self.explored_each_map[i])
                    obstacle_shared_map = np.maximum(obstacle_shared_map, self.obstacle_each_map[i])
                    target_found_shared = np.maximum(target_found_shared, self.target_found[i])
                    shared_rubble_sets[counter] = shared_rubble_sets[counter].union(agent_rubble_sets[i])
                    shared_trace_sets[counter] = shared_trace_sets[counter].union(self.agent_trace_sets[i])
                    shared_target_sets[counter] = shared_target_sets[counter].union(agent_target_sets[i])
                self.explored_shared_map.append(explored_shared_map)
                self.obstacle_shared_map.append(obstacle_shared_map)
                self.target_found_shared.append(target_found_shared)
                counter += 1
        elif self.use_full_comm:
            connected_agent_groups = [[agent_id for agent_id in range(self.num_agents)]]
        # based on agent communications, each_maps and other info are shared and target_each_maps are formed.
        if self.use_full_comm:
            for i in range(self.num_agents):
                if not self.agent_alive[i]:
                    continue
                if self.target_found.any():
                    self.target_found[i] = 1

                self.all_trace_set = self.all_trace_set.union(self.agent_trace_sets[i])
                self.all_target_set = self.all_target_set.union(agent_target_sets[i])
                self.all_rubble_set = self.all_rubble_set.union(agent_rubble_sets[i])

                self.explored_each_map[i] = explored_all_map.copy()
                self.obstacle_each_map[i] = obstacle_all_map.copy()
                self.previous_explored_each_map[i] = self.explored_each_map[i] 
            
            for i in range(self.num_agents):
                if not self.agent_alive[i]:
                    continue
                self.agent_trace_sets[i] = self.all_trace_set
                self.agent_target_sets[i] = self.all_target_set
                # self.set_target_each_map(i)
                self.agent_rubble_sets[i] = self.all_rubble_set
                # self.set_rubble_each_map(i)
        
        elif self.use_partial_comm:
            counter = 0
            for group in connected_agent_groups:
                for i in group:
                    #     if self.agent_rubble_sets[i] != None and self.rubble_cells_attended[j] != None: 
                    #         self.agent_rubble_sets[i].difference_update(self.rubble_cells_attended[j])

                    self.target_found[i] = self.target_found_shared[counter]
                    self.agent_trace_sets[i] = self.agent_trace_sets[i].union(shared_trace_sets[counter])
                    self.agent_target_sets[i] = self.agent_target_sets[i].union(shared_target_sets[counter])
                    self.set_target_each_map(i)

                    self.agent_rubble_sets[i] = self.agent_rubble_sets[i].union(shared_rubble_sets[counter])
                    self.all_rubble_set = self.all_rubble_set.union(self.agent_rubble_sets[i])
                    self.set_rubble_each_map(i)

                    self.explored_each_map[i] = self.explored_shared_map[counter]
                    self.obstacle_each_map[i] = self.obstacle_shared_map[counter]
                    self.previous_explored_each_map[i] = self.explored_each_map[i]
                    
                counter += 1
        self.set_target_all_map()
        self.set_rubble_all_map()            
        # print("agent rubble sets after comm: ", self.agent_rubble_sets)
        # print("rubble all set: ", self.all_rubble_set)

        reward_explored_all_map = explored_all_map.copy()
        reward_explored_all_map[reward_explored_all_map != 0] = 1

        reward_obstacle_all_map = obstacle_all_map.copy()
        reward_obstacle_all_map[reward_obstacle_all_map != 0] = 1

        delta_reward_all_map = reward_explored_all_map - reward_obstacle_all_map

        # reward_previous_all_map = self.previous_all_map.copy()
        # reward_previous_all_map[reward_previous_all_map != 0] = 1

        # step_reward_all_map = delta_reward_all_map - reward_previous_all_map

        # merge_explored_reward = step_reward_all_map.sum()
        
        # self.previous_all_map = explored_all_map - obstacle_all_map
        self.explored_map = np.array(explored_all_map).astype(int)[
            self.agent_view_size: self.width + self.agent_view_size, self.agent_view_size: self.width + self.agent_view_size]

        occupied_each_map = np.copy(self.obstacle_each_map)
        for i in range(self.num_agents):
            if not self.agent_alive[i]:
                continue
            for rubble_pos in self.agent_rubble_sets[i]:
                occupied_each_map[i][rubble_pos[1]+self.agent_view_size, rubble_pos[0]+self.agent_view_size] = 1
            for target_pos in self.agent_target_sets[i]:
                occupied_each_map[i][target_pos[1]+self.agent_view_size, target_pos[0]+self.agent_view_size] = 1
            # if self.target_found[i]:
            #     occupied_each_map[i][self.target_pos[1]+self.agent_view_size, self.target_pos[0]+self.agent_view_size] = 1
            
        occupied_all_map = np.copy(obstacle_all_map)
        for rubble_pos in self.all_rubble_set:
            occupied_all_map[rubble_pos[1]+self.agent_view_size, rubble_pos[0]+self.agent_view_size] = 1
        if self.is_target_found:
            occupied_all_map[self.target_pos[1]+self.agent_view_size, self.target_pos[0]+self.agent_view_size] = 1
        # print("all rubble set: ", self.all_rubble_set)
        # print("agent rubble sets: ", self.agent_rubble_sets)
        number_agent_rubbles_attended = []
        agent_number_of_known_rubbles = np.zeros(self.num_agents)
        for i in range(self.num_agents):
            number_agent_rubbles_attended.append(len(self.rubble_cells_attended[i]))
            agent_number_of_known_rubbles[i] = len(self.agent_rubble_sets[i])
        
        number_of_rubbles_removed = sum(number_agent_rubbles_attended) 
        # print("number_of_rubbles_removed: ", number_of_rubbles_removed)

        self.info = {}
        self.info['explored_all_map'] = np.array(explored_all_map)
        self.info['agent_pos'] = np.array(self.agent_pos)
        self.info['current_agent_pos'] = np.array(current_agent_pos)
        self.info['agent_direction'] = np.array(self.agent_dir)
        self.info['explored_each_map'] = np.array(self.explored_each_map)
        self.info['occupied_all_map'] = np.array(occupied_all_map)
        self.info['occupied_each_map'] = np.array(occupied_each_map)
        self.info['rubble_each_map'] = np.array(self.rubble_each_map)
        self.info['global_rubble_each_map'] = self.global_rubble_each_map
        self.info['rubble_all_map'] = np.array(self.rubble_all_map)
        self.info['global_rubble_all_map'] = self.global_rubble_all_map
        self.info['target_each_map'] = np.array(self.target_each_map)
        self.info['global_target_each_map'] = self.global_target_each_map
        self.info['target_all_map'] = np.array(self.target_all_map)
        self.info['global_target_all_map'] = self.global_target_all_map
        self.info['each_agent_trajectory_map'] = self.each_agent_trajectory_map
        self.info['target_rescued'] = self.target_rescued
        self.info['target_found'] = self.target_found
        self.info['is_target_found'] = self.is_target_found
        # self.info['agent_paths'] = self.paths
        self.info['agent_groups'] = self.agent_groups
        self.info['connected_agent_groups'] = connected_agent_groups
        self.info['agent_alive'] = self.agent_alive
        # self.info['agent_inventory'] = self.agent_inventory
        self.info['number_of_rubbles_removed'] = number_of_rubbles_removed
        self.info['agent_number_of_known_rubbles'] = agent_number_of_known_rubbles
        self.info['agent_local_views'] = agent_local_views
        self.info['agent_target_sets'] = self.agent_target_sets
        
        # self.info['agent_rubble_sets'] = self.agent_rubble_sets
        # self.info['agent_target_sets'] = self.agent_target_sets
        # self.info['agent_trace_sets'] = self.agent_trace_sets
        # self.info['all_rubble_set'] = self.all_rubble_set
        # self.info['all_target_set'] = self.all_target_set
        # self.info['all_trace_set'] = self.all_trace_set
        

        if self.target_rescued and self.stayed_switch == 0:
            self.target_rescued_step = self.num_step
            self.stayed_switch = 1
        self.info['target_rescued_step'] = self.target_rescued_step
        if self.target_found.any() and self.found_switch == 0:
            self.target_found_step = self.num_step
            self.found_switch = 1
        self.info['target_found_step'] = self.target_found_step
        
        # overlapping explored cells are not rewarded
        pure_step_reward_each_map = step_reward_each_map.copy() 
        each_agent_pure_exp_rewards = []
        for i in range(self.num_agents):
            for j in range(self.num_agents):
                if j != i:
                    pure_step_reward_each_map[i] = pure_step_reward_each_map[i] - step_reward_each_map[j]
            each_agent_pure_exp_rewards.append((pure_step_reward_each_map[i] > 0).sum())

        exp_reward_mask = np.zeros((self.num_agents))
        for i in range(self.num_agents):
            if self.agent_types_list[i] == 1 and not self.target_found[i]: # explorer agents get rewarded for exploring before finding the target
                exp_reward_mask[i] = 1
        
        exploration_reward_weight = 50/self.no_wall_size # 50 is the maximum total reward for exploration
                
        self.info['agent_explored_reward'] = np.array(each_agent_pure_exp_rewards) * exploration_reward_weight * exp_reward_mask
        # self.info['merge_explored_reward'] = merge_explored_reward * exploration_reward_weight #not used
        

        if self.use_intrinsic_reward: # exploration reward
            # Set a maximum value for this reward and clip it
            np.clip(self.info['agent_explored_reward'], 0, 20, out=self.info['agent_explored_reward'])
            reward += np.expand_dims(self.info['agent_explored_reward'], axis=1)

        self.agent_exploration_reward = self.info['agent_explored_reward']
        # self.merge_reward = self.info['merge_explored_reward']
        self.merge_ratio = delta_reward_all_map.sum() / self.no_wall_size  # (self.width * self.height)
        
        self.info['merge_explored_ratio'] = self.merge_ratio
        # self.info['agent_explored_ratio'] = self.agent_ratio
        

        # if self.num_step == 20:
        #     self.agent_alive[1] = 0
        # for i in range(self.num_agents):
        #     if self._rand_float(0, 1) < 0.01:
        #         self.agent_alive[i] = 0

        if self.num_step >= self.max_steps:
            done = True
        return obs, reward, done, self.info

    def ft_get_short_term_goals(self, args, mode=""):
        '''
        frontier-based methods compute actions
        '''
        # self.info = self.ft_info
        replan = [False for _ in range(self.num_agents)]
        if self.use_full_comm:
            current_agent_pos = self.info["current_agent_pos"]
            location_lists = []
            for agent_id in range(self.num_agents):
                location_lists.append(current_agent_pos)
        elif self.use_partial_comm: #only the position of connected agents are known to each agent
            current_agent_pos = self.info["current_agent_pos"]
            connected_agent_groups = get_connected_agents(self.agent_groups) # connected agents encompass agent groups
            location_lists = []
            for agent_id in range(self.num_agents):
                location_list = []
                for group in connected_agent_groups:
                    if agent_id in group:
                        for agent in group:
                            location_list.append(current_agent_pos[agent])
                location_lists.append(location_list)

        goals = [None for _ in range(self.num_agents)]
        for agent_id in range(self.num_agents):
            if self.use_full_comm:
                explored = (self.info['explored_all_map'] > 0).astype(np.int32)
                obstacle = (self.info['occupied_all_map'] > 0).astype(np.int32)
                obstacle_reduced = (self.info['occupied_each_map'][agent_id] > 0).astype(np.int32)[
                    self.agent_view_size:self.agent_view_size+self.width, self.agent_view_size:self.agent_view_size+self.height]
                if self.use_agent_obstacle:
                    for a in range(self.num_agents):
                        if a != agent_id:
                            obstacle[current_agent_pos[a][0], current_agent_pos[a][1]] = 1
            elif self.use_partial_comm:
                explored = (self.info['explored_each_map'][agent_id] > 0).astype(np.int32)
                obstacle = (self.info['occupied_each_map'][agent_id] > 0).astype(np.int32)
                obstacle_reduced = (self.info['occupied_each_map'][agent_id] > 0).astype(np.int32)[
                    self.agent_view_size:self.agent_view_size+self.width, self.agent_view_size:self.agent_view_size+self.height]
                if self.use_agent_obstacle:
                    for group in connected_agent_groups:
                        if agent_id in group:
                            for a in group:
                                if a != agent_id:
                                    obstacle[current_agent_pos[a][0], current_agent_pos[a][1]] = 1
            else:
                raise NotImplementedError

            H, W = explored.shape
            steps = [(-1, 0), (1, 0), (0, -1), (0, 1)]
            map = np.ones((H, W)).astype(np.int32) * 3  # 3 for unknown area
            map[explored == 1] = 0  # 0 for explored area
            map[obstacle == 1] = 1  # 1 for obstacles
            # Set frontiers.
            for x in range(H):
                for y in range(W):
                    if map[x, y] == 0:
                        neighbors = [(x+dx, y+dy) for dx, dy in steps]
                        if sum([(map[u, v] == 3) for u, v in neighbors]) > 0:
                            map[x, y] = 2  # 2 for targets (frontiers)
            map[:self.agent_view_size, :] = 1
            map[H-self.agent_view_size:, :] = 1
            map[:, :self.agent_view_size] = 1
            map[:, W-self.agent_view_size:] = 1
            unexplored = (map == 3).astype(np.int32)
            map[map == 3] = 0  # set unknown area to explorable
            # print(map)
            if self.num_step >= 1:
                # print("ft goals for agent ", agent_id, " is: ", self.ft_goals[agent_id])
                # print("map value for agent ", agent_id, " is: ", map[self.ft_goals[agent_id][0], self.ft_goals[agent_id][1]])
                # print("current agent pos: ", current_agent_pos[agent_id])
                # print("unexplored value for agent ", agent_id, " is: ", unexplored[self.ft_goals[agent_id][0], self.ft_goals[agent_id][1]])
                if (map[self.ft_goals[agent_id][0], self.ft_goals[agent_id][1]] != 2) and\
                (unexplored[self.ft_goals[agent_id][0], self.ft_goals[agent_id][1]] == 0):
                    replan[agent_id] = True
                if current_agent_pos[agent_id][0] == self.ft_goals[agent_id][0] and current_agent_pos[agent_id][1] == self.ft_goals[agent_id][1]:
                    replan[agent_id] = True
                    map[self.ft_goals[agent_id][0], self.ft_goals[agent_id][1]] = 0
                    # print("replanning is triggered")
                    
            if replan[agent_id] or self.ft_goals[agent_id] is None:
                if mode == 'apf':
                    apf = APF(args)
                    path = apf.schedule(map, location_lists[agent_id], steps,
                                        current_agent_pos[agent_id], self.apf_penalty[agent_id])
                    goal = path[-1]
                elif mode == 'utility':
                    goal = utility_goal(map, unexplored, current_agent_pos[agent_id], steps)
                elif mode == 'nearest':
                    goal = nearest_goal(map, current_agent_pos[agent_id], steps)
                elif mode == 'rrt':
                    goal = rrt_goal(map, unexplored, current_agent_pos[agent_id])
                elif mode == 'voronoi':
                    goal = voronoi_goal(map, unexplored, current_agent_pos, agent_id, steps)
                goals[agent_id] = goal
            else:
                goals[agent_id] = self.ft_goals[agent_id]

            if self.agent_types_list[agent_id] == 0 and self.target_found[agent_id] == 1:
                target_pos = [self.target_pos[0], self.target_pos[1]]
                target_neighbors = [(target_pos[0]+dx, target_pos[1]+dy) for dx, dy in steps]
                agent_pos = self.agent_pos[agent_id]

                obs_list = []
                for x in range(self.width):
                    #enclosing the map for the path planner
                    obs_list.append((x,-1))
                    obs_list.append((x,self.height))
                    for y in range(self.height):
                        if obstacle_reduced[x, y] == 1:
                            obs_list.append((y, x))
                for y in range(self.height):
                    obs_list.append((-1,y))
                    obs_list.append((self.width,y))
                
                shortest_path = float('inf')
                for goal in target_neighbors:
                    if obstacle_reduced[goal[1], goal[0]] == 1:
                        continue
                    path_planner = astar.AStar(obs_list, tuple(agent_pos), tuple(goal), "manhattan")
                    path, _ = path_planner.searching()
                    if path is None:
                        continue
                    if len(path) != 1 and len(path) < shortest_path:
                        shortest_path = len(path)
                        goals[agent_id] = [goal[1]+self.agent_view_size, goal[0]+self.agent_view_size]
                if shortest_path == float('inf'):
                    goals[agent_id] = self.ft_goals[agent_id]
        
        
        self.ft_goals = goals.copy()
        # print("goals are: ", goals)
        return goals

    def ft_get_short_term_actions(self,
                                  goals,
                                  mode,
                                  radius
                                  ): #lets see if this is needed or we can just use get_short_term_actions
        # self.info = self.ft_info
        actions = []
        current_agent_pos = self.info["current_agent_pos"]
        for agent_id in range(self.num_agents):
            if self.use_full_comm:
                explored = (self.info['explored_all_map'] > 0).astype(np.int32)
                obstacle = (self.info['occupied_all_map'] > 0).astype(np.int32)
            elif self.use_partial_comm:
                explored = (self.info['explored_each_map'][agent_id] > 0).astype(np.int32)
                obstacle = (self.info['occupied_each_map'][agent_id] > 0).astype(np.int32)
            else:
                raise NotImplementedError
            
            if self.use_agent_obstacle:
                for a in range(self.num_agents):
                    if a != agent_id:
                        obstacle[current_agent_pos[a][0], current_agent_pos[a][1]] = 1

            H, W = explored.shape
            map = np.ones((H, W)).astype(np.int32) * 3  # 3 for unknown area
            map[explored == 1] = 0  # 0 for explored area
            map[obstacle == 1] = 1  # 1 for obstacles
            map[:self.agent_view_size, :] = 1
            map[H-self.agent_view_size:, :] = 1
            map[:, :self.agent_view_size] = 1
            map[:, W-self.agent_view_size:] = 1
            # Set unexplored.
            unexplored = (map == 3).astype(np.int32)
            # Initialize cost map.
            temp_map = map.copy().astype(np.float32)
            temp_map[map != 1] = 1  # free & frontiers & unknown
            temp_map[map == 1] = np.inf  # obstacles

            if mode == 'normal':
                pass
            elif mode == 'utility':
                # cost = 1 - unexplored (%)
                H, W = map.shape
                for x in range(H):
                    for y in range(W):
                        if map[x, y] == 1:
                            temp_map[x, y] = np.inf
                        else:
                            utility = unexplored[x-radius:x+radius+1, y-radius:y +
                                                radius+1].sum() / (math.pow(radius*2+1, 2))
                            temp_map[x, y] = 1.0 + (1.0 - utility) * 2.0
            else:
                raise NotImplementedError

            goal = [goals[agent_id][0], goals[agent_id][1]]
            agent_pos = [current_agent_pos[agent_id][0], current_agent_pos[agent_id][1]]
            agent_dir = self.agent_dir[agent_id]
            # path = pyastar2d.astar_path(temp_map, agent_pos, goal, allow_diagonal=False) # old astar method
            
            obs_list = []
            for x in range(self.width):
                #enclosing the map for the path planner
                obs_list.append((x,-1))
                obs_list.append((x,self.height))
                for y in range(self.height):
                    if obstacle[x, y] == 1:
                        obs_list.append((x, y))
            for y in range(self.height):
                obs_list.append((-1,y))
                obs_list.append((self.width,y))
            
            path_planner = astar.AStar(obs_list, tuple(agent_pos), tuple(goal), "manhattan")
            path, _ = path_planner.searching()
            
            path = path[::-1]

            if len(path) == 2 and path[0] == path[1]:
                # if inputs[agent_id][2] == 1:
                #     actions.append(4) # interact
                # else:
                actions.append(3) # stop, goal is reached
                continue
            if len(path) == 1:
                actions.append(4) #goal is unreachable
                continue

            
            relative_pos = np.array(path[1]) - np.array(agent_pos)
            action = self.relative_pose2action(agent_dir, relative_pos)
            actions.append(action)

        return actions
    
    def relative_pose2action(self, agent_dir, relative_pos):
        # first quadrant
        if relative_pos[0] < 0 and relative_pos[1] > 0:
            if agent_dir == 0 or agent_dir == 3:
                return 2  # forward
            if agent_dir == 1:
                return 0  # turn left
            if agent_dir == 2:
                return 1  # turn right
        # second quadrant
        if relative_pos[0] > 0 and relative_pos[1] > 0:
            if agent_dir == 0 or agent_dir == 1:
                return 2  # forward
            if agent_dir == 2:
                return 0  # turn left
            if agent_dir == 3:
                return 1  # turn right
        # third quadrant
        if relative_pos[0] > 0 and relative_pos[1] < 0:
            if agent_dir == 1 or agent_dir == 2:
                return 2  # forward
            if agent_dir == 3:
                return 0  # turn left
            if agent_dir == 0:
                return 1  # turn right
        # fourth quadrant
        if relative_pos[0] < 0 and relative_pos[1] < 0:
            if agent_dir == 2 or agent_dir == 3:
                return 2  # forward
            if agent_dir == 0:
                return 0  # turn left
            if agent_dir == 1:
                return 1  # turn right
        if relative_pos[0] == 0 and relative_pos[1] == 0:
            # turn around
            return 1
        if relative_pos[0] == 0 and relative_pos[1] > 0:
            if agent_dir == 0:
                return 2
            if agent_dir == 1:
                return 0
            else:
                return 1
        if relative_pos[0] == 0 and relative_pos[1] < 0:
            if agent_dir == 2:
                return 2
            if agent_dir == 1:
                return 1
            else:
                return 0
        if relative_pos[0] > 0 and relative_pos[1] == 0:
            if agent_dir == 1:
                return 2
            if agent_dir == 0:
                return 1
            else:
                return 0
        if relative_pos[0] < 0 and relative_pos[1] == 0:
            if agent_dir == 3:
                return 2
            if agent_dir == 0:
                return 0
            else:
                return 1
        return None
    
    def distance(self,pos_a,pos_b):
        dis = np.square(pos_a[0]-pos_b[0])+np.square(pos_a[1]-pos_b[1])
        return dis

    def get_short_term_action(self, inputs):
        actions = []
        paths = []
        for agent_id in range(self.num_agents):
            if self.use_full_comm: 
                explored = (self.info['explored_all_map'] > 0).astype(np.int32)[
                    self.agent_view_size:self.agent_view_size+self.width, self.agent_view_size:self.agent_view_size+self.height]
                obstacle = (self.info['occupied_all_map'] > 0).astype(np.int32)[
                    self.agent_view_size:self.agent_view_size+self.width, self.agent_view_size:self.agent_view_size+self.height]
            elif self.use_partial_comm:
                explored = (self.info['explored_each_map'][agent_id] > 0).astype(np.int32)[
                    self.agent_view_size:self.agent_view_size+self.width, self.agent_view_size:self.agent_view_size+self.height]
                obstacle = (self.info['occupied_each_map'][agent_id] > 0).astype(np.int32)[
                    self.agent_view_size:self.agent_view_size+self.width, self.agent_view_size:self.agent_view_size+self.height]
            else:
                raise NotImplementedError
            
            if self.use_agent_obstacle:
                for a in range(self.num_agents):
                    if a != agent_id:
                        obstacle[self.agent_pos[a][1], self.agent_pos[a][0]] = 1
            goal = [int(inputs[agent_id][1]), int(inputs[agent_id][0])]

            agent_pos = [self.agent_pos[agent_id][1], self.agent_pos[agent_id][0]]
            agent_dir = self.agent_dir[agent_id]
            
            obs_list = []
            for x in range(self.width):
                #enclosing the map for the path planner
                obs_list.append((x,-1))
                obs_list.append((x,self.height))
                for y in range(self.height):
                    if obstacle[x, y] == 1:
                        obs_list.append((x, y))
            for y in range(self.height):
                obs_list.append((-1,y))
                obs_list.append((self.width,y))

            
            # path_planner = dstarlite.DStar(obs_list, tuple(agent_pos), tuple(goal), "manhattan")
            # path_planner.ComputePath()
            # path = path_planner.extract_path()
            path_planner = astar.AStar(obs_list, tuple(agent_pos), tuple(goal), "manhattan")
            path, _ = path_planner.searching()
            
            path = path[::-1]
            
            paths.append(path)

            # path_cost = path_planner.cost(tuple(agent_pos), tuple(goal))
            
            if len(path) == 2 and path[0] == path[1]:
                # if inputs[agent_id][2] == 1:
                #     actions.append(4) # interact
                # else:
                actions.append(3) # stop, goal is reached
                continue
            if len(path) == 1:
                
                actions.append(4) #goal is unreachable
                continue
            relative_pos = np.array(path[1]) - np.array(agent_pos)
            
            
            # first quadrant
            if relative_pos[0] < 0 and relative_pos[1] > 0:
                if agent_dir == 0 or agent_dir == 3:
                    actions.append(2)  # forward
                    continue
                if agent_dir == 1:
                    actions.append(0)  # turn left
                    continue
                if agent_dir == 2:
                    actions.append(1)  # turn right
                    continue
            # second quadrant
            if relative_pos[0] > 0 and relative_pos[1] > 0:
                if agent_dir == 0 or agent_dir == 1:
                    actions.append(2)  # forward
                    continue
                if agent_dir == 2:
                    actions.append(0)  # turn left
                    continue
                if agent_dir == 3:
                    actions.append(1)  # turn right
                    continue
            # third quadrant
            if relative_pos[0] > 0 and relative_pos[1] < 0:
                if agent_dir == 1 or agent_dir == 2:
                    actions.append(2)  # forward
                    continue
                if agent_dir == 3:
                    actions.append(0)  # turn left
                    continue
                if agent_dir == 0:
                    actions.append(1)  # turn right
                    continue
            # fourth quadrant
            if relative_pos[0] < 0 and relative_pos[1] < 0:
                if agent_dir == 2 or agent_dir == 3:
                    actions.append(2)  # forward
                    continue
                if agent_dir == 0:
                    actions.append(0)  # turn left
                    continue
                if agent_dir == 1:
                    actions.append(1)  # turn right
                    continue
            if relative_pos[0] == 0 and relative_pos[1] == 0:
                # turn around
                actions.append(1)
                continue
            if relative_pos[0] == 0 and relative_pos[1] > 0:
                if agent_dir == 0:
                    actions.append(2)
                    continue
                if agent_dir == 1:
                    actions.append(0)
                    continue
                if agent_dir == 2:
                    if self._rand_float(0, 1) < 0.5:
                        actions.append(1)
                    else:
                        actions.append(0)
                else:
                    actions.append(1)
                    continue
            if relative_pos[0] == 0 and relative_pos[1] < 0:
                if agent_dir == 2:
                    actions.append(2)
                    continue
                if agent_dir == 1:
                    actions.append(1)
                    continue
                if agent_dir == 0:
                    if self._rand_float(0, 1) < 0.5:
                        actions.append(1)
                    else:
                        actions.append(0)
                else:
                    actions.append(0)
                    continue
            if relative_pos[0] > 0 and relative_pos[1] == 0:
                if agent_dir == 1:
                    actions.append(2)
                    continue
                if agent_dir == 0:
                    actions.append(1)
                    continue
                if agent_dir == 3:
                    if self._rand_float(0, 1) < 0.5:
                        actions.append(1)
                    else:
                        actions.append(0)
                else:
                    actions.append(0)
                    continue
            if relative_pos[0] < 0 and relative_pos[1] == 0:
                if agent_dir == 3:
                    actions.append(2)
                    continue
                if agent_dir == 0:
                    actions.append(0)
                    continue
                if agent_dir == 1:
                    if self._rand_float(0, 1) < 0.5:
                        actions.append(1)
                    else:
                        actions.append(0)
                else:
                    actions.append(1)
                    continue
    
        # self.paths = paths
        # print("agent 1s paths is ", paths[0])
        return actions
    
    def get_available_actions(self, act_dim):
        # available_actions = np.ones((self.num_agents, *act_dim))
        # for i in range(self.num_agents):
                           
        #     occupied = self.info['occupied_each_map'][i][self.agent_view_size:self.full_w - self.agent_view_size,
        #                                                 self.agent_view_size:self.full_h - self.agent_view_size]
        #     occupied = occupied.T

        #     # if self.agent_types_list[i] == 0 and self.target_found[i]: #actuator agents
        #     #         target_adjacent_cells = self.adjacent_cells(*self.target_pos)

        #     for x in range(-self.action_size, self.action_size+1):
        #         for y in range(-self.action_size, self.action_size+1):
        #             coord = np.array([x, y]) + self.agent_pos[i]
        #             # if self.agent_types_list[i] == 0 and self.target_found[i]: # actuator agents can only move to the target if it is found
        #             #     if tuple(coord) in target_adjacent_cells:
        #             #         available_actions[i, x + self.action_size, y + self.action_size] = 1
        #             if coord[0] < 0 or coord[0] >= self.width or coord[1] < 0 or coord[1] >= self.height: #map boundaries
        #                 available_actions[i, x + self.action_size, y + self.action_size] = 0
        #             elif occupied[coord[0], coord[1]] == 1: #Moving to occupied space is not available
        #                 available_actions[i, x + self.action_size, y + self.action_size] = 0
        available_actions = np.ones((self.num_agents, act_dim))
        for i in range(self.num_agents):
                           
            occupied = self.info['occupied_each_map'][i][self.agent_view_size:self.full_w - self.agent_view_size,
                                                        self.agent_view_size:self.full_h - self.agent_view_size]
            occupied = occupied.T
            explored = self.info['explored_each_map'][i][self.agent_view_size:self.full_w - self.agent_view_size,
                                                        self.agent_view_size:self.full_h - self.agent_view_size]
            explored = explored.T
            # obs_list = []
            # for x in range(self.width):
            #     #enclosing the map for the path planner
            #     obs_list.append((x,-1))
            #     obs_list.append((x,self.height))
            #     for y in range(self.height):
            #         if occupied[x, y] == 1:
            #             obs_list.append((x, y))
            # for y in range(self.height):
            #     obs_list.append((-1,y))
            #     obs_list.append((self.width,y))
        
            for x in range(-self.action_size, self.action_size+1):
                for y in range(-self.action_size, self.action_size+1):
                    coord = np.array([x, y]) + self.agent_pos[i]
                    if coord[0] < 1 or coord[0] >= self.width - 1 or coord[1] < 1 or coord[1] >= self.height - 1: # Outside map boundaries and edges of the map
                        available_actions[i, (x + self.action_size)*(2*self.action_size+1) + (y + self.action_size)] = 0
                    elif occupied[coord[0], coord[1]] == 1: # Moving to occupied space is not available
                        available_actions[i, (x + self.action_size)*(2*self.action_size+1) + (y + self.action_size)] = 0
                    elif explored[coord[0], coord[1]] == 0: # Moving to unexplored space is not available
                        available_actions[i, (x + self.action_size)*(2*self.action_size+1) + (y + self.action_size)] = 0
                    else: 
                        if self.use_agent_obstacle:
                            for j in range(self.num_agents):
                                if j != i:
                                    if coord[0] == self.agent_pos[j][0] and coord[1] == self.agent_pos[j][1]:
                                        available_actions[i, (x + self.action_size)*(2*self.action_size+1) + (y + self.action_size)] = 0
                                        break
                        for adjacent_cells in self.adjacent_cells(coord[0], coord[1]): # If the cell is surrounded by obstacles, it is not available
                            if occupied[adjacent_cells[0], adjacent_cells[1]] == 0:
                                break
                        else:
                            available_actions[i, (x + self.action_size)*(2*self.action_size+1) + (y + self.action_size)] = 0

                        
        # print("available action are chosen")
        # available_actions = np.ones((self.num_agents, *act_dim))
        return available_actions