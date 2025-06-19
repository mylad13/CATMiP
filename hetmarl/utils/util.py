import glob
import os
import numpy as np
import math
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class AsynchControl:
    def __init__(self, num_envs, num_agents, limit, random_fn, min_wait, max_wait, rest_time):
        self.num_envs = num_envs
        self.num_agents = num_agents
        self.limit = limit
        self.random_fn = random_fn
        self.min_wait = min_wait
        self.max_wait = max_wait
        self.rest_time = rest_time
        self.reset()
    
    def reset(self):
        self.cnt = np.zeros((self.num_envs, self.num_agents), dtype=np.int32)
        self.rest = np.zeros((self.num_envs, self.num_agents), dtype=np.int32)
        self.wait = np.zeros((self.num_envs, self.num_agents), dtype=np.int32)
        self.active = np.ones((self.num_envs, self.num_agents), dtype=np.int32)
        self.standby = np.zeros((self.num_envs, self.num_agents), dtype=np.int32)
        for e in range(self.num_envs):
            for a in range(self.num_agents):
                self.rest[e, a] = self.rest_time
                # self.rest[e, a] = self.random_fn(18,21)
                self.wait[e, a] = self.random_fn(self.min_wait,self.max_wait)
    
    def step(self):
        for e in range(self.num_envs):
            for a in range(self.num_agents):
                self.rest[e, a] -= 1
                self.active[e, a] = 0
                if self.standby[e,a]:
                    self.wait[e, a] -= 1
                    if self.wait[e, a] <= 0:
                        self.activate(e, a)
                if self.rest[e, a] <= 0:
                    if self.cnt[e, a] < self.limit:
                        self.standby[e, a] = 1
                        # self.activate(e, a)
                        

    def activate(self, e, a):
        self.cnt[e, a] += 1
        self.active[e, a] = 1
        self.standby[e, a] = 0
        self.wait[e, a] = self.random_fn(self.min_wait,self.max_wait)
        # self.wait[e, a] = 0
        # self.rest[e, a] = self.random_fn(180,210)
        self.rest[e, a] = self.rest_time

    def active_agents(self):
        ret = []
        for e in range(self.num_envs):
            for a in range(self.num_agents):
                if self.active[e, a]:
                    ret.append((e, a, self.cnt[e, a]))
        return ret
    
    def active_agents_threads(self):
        ret = [] # this first item in the list is the thread number, and the following items are the agent numbers
        for e in range(self.num_envs):
            thread = []
            thread.append(e)
            for a in range(self.num_agents):
                if self.active[e, a]:
                    thread.append(a)
            ret.append(thread)
        return ret
    
    def standby_agents(self):
        ret = [] # this first item in the list is the thread number, and the following items are the agent numbers
        for e in range(self.num_envs):
            thread = []
            thread.append(e)
            for a in range(self.num_agents):
                if self.standby[e, a]:
                    thread.append(a)
            ret.append(thread)
        return ret

def check(input):
    if type(input) == np.ndarray:
        return torch.from_numpy(input)
        
def get_gard_norm(it):
    sum_grad = 0
    for x in it:
        if x.grad is None:
            continue
        sum_grad += x.grad.norm() ** 2
    return math.sqrt(sum_grad)

def update_linear_schedule(optimizer, epoch, total_num_epochs, initial_lr):
    """Decreases the learning rate linearly"""
    lr = initial_lr - (initial_lr * (epoch / float(total_num_epochs)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def huber_loss(e, d):
    a = (abs(e) <= d).float()
    b = (e > d).float()
    return a*e**2/2 + b*d*(abs(e)-d/2)

def mse_loss(e):
    return e**2/2

def get_shape_from_obs_space(obs_space):
    if obs_space.__class__.__name__ == 'Box':
        obs_shape = obs_space.shape
    elif obs_space.__class__.__name__ == 'list':
        obs_shape = obs_space
    elif obs_space.__class__.__name__ == 'Dict':
        obs_shape = obs_space.spaces
    else:
        raise NotImplementedError
    return obs_shape

def get_shape_from_act_space(act_space):
    if act_space.__class__.__name__ == 'Discrete':
        act_shape = 1
    elif act_space.__class__.__name__ == "MultiDiscrete":
        act_shape = act_space.shape
    elif act_space.__class__.__name__ == "Box":
        act_shape = act_space.shape[0]
    elif act_space.__class__.__name__ == "MultiBinary":
        act_shape = act_space.shape[0]
    else:  # agar
        act_shape = act_space[0].shape[0] + 1  
    return act_shape

# def get_shape_from_act_space(act_space):
#     if act_space.__class__.__name__ == 'Discrete':
#         act_shape = 1
#     elif act_space.__class__.__name__ == "MultiDiscrete":
#         act_shape = act_space.high - act_space.low + 1 # this would be the grid size (e.g, (25,25))
#     elif act_space.__class__.__name__ == "Box":
#         act_shape = act_space.shape[0]
#     elif act_space.__class__.__name__ == "MultiBinary":
#         act_shape = act_space.shape[0]
#     else:  # agar
#         act_shape = act_space[0].shape[0] + 1
#     act_num = len(act_shape)
#     return act_shape, act_num

def tile_images(img_nhwc):
    """
    Tile N images into one big PxQ image
    (P,Q) are chosen to be as close as possible, and if N
    is square, then P=Q.
    input: img_nhwc, list or array of images, ndim=4 once turned into array
        n = batch index, h = height, w = width, c = channel
    returns:
        bigim_HWc, ndarray with ndim=3
    """
    img_nhwc = np.asarray(img_nhwc)
    N, h, w, c = img_nhwc.shape
    H = int(np.ceil(np.sqrt(N)))
    W = int(np.ceil(float(N)/H))
    img_nhwc = np.array(list(img_nhwc) + [img_nhwc[0]*0 for _ in range(N, H*W)])
    img_HWhwc = img_nhwc.reshape(H, W, h, w, c)
    img_HhWwc = img_HWhwc.transpose(0, 2, 1, 3, 4)
    img_Hh_Ww_c = img_HhWwc.reshape(H*h, W*w, c)
    return img_Hh_Ww_c

def get_connected_agents(matrix):
            def dfs(matrix, node, visited, connected_nodes): #depth first search
                n = len(matrix)
                for neighbor in range(n):
                    if matrix[node][neighbor] == 1 and neighbor not in visited:
                        connected_nodes.append(neighbor)
                        visited.add(neighbor)
                        dfs(matrix, neighbor, visited, connected_nodes)
            n = len(matrix)
            visited = set()
            connected_agents = []

            for node in range(n):
                if node not in visited:
                    connected_nodes = []
                    dfs(matrix, node, visited, connected_nodes)
                    connected_agents.append(connected_nodes)

            return connected_agents

def plot_macro_obs(macro_obs, agent_num):
    """Plot all the channels of the macro observation space."""

    if macro_obs == None:
        print("No macro observation available.")
        return
    global_exploration = macro_obs['global_agent_map'][0, agent_num, 0]
    global_occupancy = macro_obs['global_agent_map'][0, agent_num, 1]
    global_target = macro_obs['global_agent_map'][0, agent_num, 2]
    global_ego_pose = macro_obs['global_agent_map'][0, agent_num, 3]
    global_rescuers = macro_obs['global_agent_map'][0, agent_num, 4]
    global_explorers = macro_obs['global_agent_map'][0, agent_num, 5]
    global_goal = macro_obs['global_agent_map'][0, agent_num, 6]

    local_exploration = macro_obs['local_agent_map'][0, agent_num, 0]
    local_occupancy = macro_obs['local_agent_map'][0, agent_num, 1]
    local_target = macro_obs['local_agent_map'][0, agent_num, 2]
    local_rescuers = macro_obs['local_agent_map'][0, agent_num, 3]
    local_explorers = macro_obs['local_agent_map'][0, agent_num, 4]
    local_goal = macro_obs['local_agent_map'][0, agent_num, 5]

    agent_indication = macro_obs['agent_class_identifier'][0, agent_num].reshape(1, -1)
    
    fig, axs = plt.subplots(2, 7, figsize=(16, 8))
    fig.suptitle('Macro Observation Space', fontsize=16)

    # Global maps
    axs[0, 0].imshow(global_exploration, cmap='gray')
    axs[0, 0].set_title('Global Exploration')
    axs[0, 1].imshow(global_occupancy, cmap='gray')
    axs[0, 1].set_title('Global Occupancy')
    axs[0, 2].imshow(global_target, cmap='gray')
    axs[0, 2].set_title('Global Target')
    axs[0, 3].imshow(global_ego_pose, cmap='gray')
    axs[0, 3].set_title('Global Ego Pose')
    axs[0, 4].imshow(global_rescuers, cmap='gray')
    axs[0, 4].set_title('Global Rescuers')
    axs[0, 5].imshow(global_explorers, cmap='gray')
    axs[0, 5].set_title('Global Explorers')
    axs[0, 6].imshow(global_goal, cmap='gray')
    axs[0, 6].set_title('Global Goal')

    # Local maps
    axs[1, 0].imshow(local_exploration, cmap='gray')
    axs[1, 0].set_title('Local Exploration')
    axs[1, 1].imshow(local_occupancy, cmap='gray')
    axs[1, 1].set_title('Local Occupancy')
    axs[1, 2].imshow(local_target, cmap='gray')
    axs[1, 2].set_title('Local Target')
    axs[1, 3].imshow(local_rescuers, cmap='gray')
    axs[1, 3].set_title('Local Rescuers')
    axs[1, 4].imshow(local_explorers, cmap='gray')
    axs[1, 4].set_title('Local Explorers')
    axs[1, 5].imshow(local_goal, cmap='gray')
    axs[1, 5].set_title('Local Goal')
    axs[1, 6].imshow(agent_indication, cmap='gray')
    axs[1, 6].set_title('Agent Class Identifier')

    # Hide unused subplots
    for i in range(6, 7):
        axs[1, i].axis('off')

    # Adjust layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
