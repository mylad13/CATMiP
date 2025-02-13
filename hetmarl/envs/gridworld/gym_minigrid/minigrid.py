import math
import hashlib
import gym
from enum import IntEnum
import numpy as np
from gym import error, spaces, utils
from gym.utils import seeding
from .rendering import *
import matplotlib.pyplot as plt
from functools import reduce
import time

from hetmarl.utils import astar

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def bresenham_line(x1, y1, x2, y2):
    """Compute the set of cells that lie on a line between (x1, y1) and (x2, y2)
    using Bresenham's line algorithm."""
    cells = []
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    sx = -1 if x1 > x2 else 1
    sy = -1 if y1 > y2 else 1
    err = dx - dy
    while x1 != x2 or y1 != y2:
        cells.append((x1, y1))
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x1 += sx
        if e2 < dx:
            err += dx
            y1 += sy
    cells.append((x2, y2))
    return cells

def check_visibility(cells, grid):
    visible_cells = []
    for cell in cells:
        obj = grid.get(cell[0], cell[1])
        if obj is not None and obj.see_behind() == False:
            visible_cells.append(cell)
            break #discard cells after any obstacles
        visible_cells.append(cell)
    return visible_cells

def l1distance(x, y):
    return abs(x[0] - y[0]) + abs(x[1] - y[1])

def euclideandistance(x, y):
    return math.sqrt((x[0] - y[0])**2 + (x[1] - y[1])**2)
# Size in pixels of a tile in the full-scale human view
TILE_PIXELS = 32

# Map of color names to RGB values
COLORS = {
    'red': np.array([255, 0, 0]),
    'green': np.array([0, 255, 0]),
    'blue': np.array([0, 0, 255]),
    'purple': np.array([112, 39, 195]),
    'yellow': np.array([255, 255, 0]),
    'grey': np.array([100, 100, 100]),
    'white': np.array([255, 255, 255]),
    'pink': np.array([255, 105, 180]),
}

COLOR_NAMES = sorted(list(COLORS.keys()))

# Used to map colors to integers
COLOR_TO_IDX = {
    'red': 0,
    'green': 1,
    'blue': 2,
    'purple': 3,
    'yellow': 4,
    'grey': 5,
    'white': 6,
    'pink': 7
}

IDX_TO_COLOR = dict(zip(COLOR_TO_IDX.values(), COLOR_TO_IDX.keys()))

# Map of object type to integers
OBJECT_TO_IDX = {
    'unseen': 0,
    'empty': 1,
    'wall': 2,
    'floor': 13,
    'door': 4,
    'key': 5,
    'ball': 6,
    'box': 7,
    'obstacle': 8,
    'target': 9,
    'lava': 10,
    'agent': 11,
    'rubble': 12,
    'trace': 3,
}

IDX_TO_OBJECT = dict(zip(OBJECT_TO_IDX.values(), OBJECT_TO_IDX.keys()))

# Map of state names to integers
STATE_TO_IDX = {
    'open': 0,
    'closed': 1,
    'locked': 2,
}

# Map of agent direction indices to vectors
DIR_TO_VEC = [
    # Pointing right (positive X)
    np.array((1, 0)),
    # Down (positive Y)
    np.array((0, 1)),
    # Pointing left (negative X)
    np.array((-1, 0)),
    # Up (negative Y)
    np.array((0, -1)),
]

FILL_COLORS = [(255, 0, 0),
               (0, 0, 255),
               (255, 255, 0),
               (0, 255, 255),
               (255, 0, 255),  
               (128, 0, 0), 
               (0, 128, 0), 
               (0, 0, 128), 
               (128, 128, 0),
               (128, 0, 128),
               (0, 128, 128),
               (0, 255, 0)]

def reject_near_doors(env, pos):
    """
    Function to filter out object positions that are near doorways
    """
    if tuple(pos) in env.doorways_adjacent_cells:
        return True
    return False
def accept_near_doors(env, pos):
    """
    Function to accept object positions that are near doorways
    """
    if tuple(pos) in env.doorways_adjacent_cells:
        return False
    return True

class WorldObj:
    """
    Base class for grid world objects
    """

    def __init__(self, type, color):
        assert type in OBJECT_TO_IDX, type
        assert color in COLOR_TO_IDX, color
        self.type = type
        self.color = color
        self.contains = None

        # Initial position of the object
        self.init_pos = None

        # Current position of the object
        self.cur_pos = None

    def can_overlap(self):
        """Can the agent overlap with this?"""
        return False

    def can_pickup(self):
        """Can the agent pick this up?"""
        return False

    def can_contain(self):
        """Can this contain another object?"""
        return False

    def see_behind(self):
        """Can the agent see behind this object?"""
        return True

    def toggle(self, env, pos):
        """Method to trigger/toggle an action this object performs"""
        return False

    def encode(self):
        """Encode the a description of this object as a 3-tuple of integers"""
        return (OBJECT_TO_IDX[self.type] * 20, COLOR_TO_IDX[self.color] * 20, 0)

    @staticmethod
    def decode(type_idx, color_idx, state):
        """Create an object from a 3-tuple state description"""

        obj_type = IDX_TO_OBJECT[type_idx]
        color = IDX_TO_COLOR[color_idx]

        if obj_type == 'empty' or obj_type == 'unseen':
            return None

        # State, 0: open, 1: closed, 2: locked
        is_open = state == 0
        is_locked = state == 2

        if obj_type == 'wall':
            v = Wall(color)
        elif obj_type == 'floor':
            v = Floor(color)
        elif obj_type == 'ball':
            v = Ball(color)
        elif obj_type == 'key':
            v = Key(color)
        elif obj_type == 'box':
            v = Box(color)
        elif obj_type == 'obstacle':
            v = Obstacle(color)
        elif obj_type == 'door':
            v = Door(color, is_open, is_locked)
        elif obj_type == 'target':
            v = Goal()
        elif obj_type == 'lava':
            v = Lava()
        elif obj_type == 'rubble':
            v = Rubble()
        elif obj_type == 'trace':
            v = Trace()
        else:
            assert False, "unknown object type in decode '%s'" % obj_type

        return v

    def render(self, r):
        """Draw this object with the given renderer"""
        raise NotImplementedError


class Goal(WorldObj): # Not to be confused with short term goals
    def __init__(self):
        super().__init__('target', 'green')

    def can_overlap(self):
        return False

    def render(self, img):
        fill_coords(img, point_in_rect(0, 1, 0, 1), COLORS[self.color])

class Trace(WorldObj):
    def __init__(self, intensity=1.0):
        super().__init__('trace', 'pink')
        self.intensity = intensity # can be used to differentiate trace tiles based on distance to target


    def can_overlap(self):
        return True

    def render(self, img):
        fill_coords(img, point_in_rect(0, 1, 0, 1), np.array([255, 255-140*self.intensity, 255-75*self.intensity]))

class Floor(WorldObj):
    """
    Colored floor tile the agent can walk over
    """

    def __init__(self, color='blue'):
        super().__init__('floor', color)

    def can_overlap(self):
        return True

    def render(self, img):
        # Give the floor a pale color
        color = COLORS[self.color] / 2
        fill_coords(img, point_in_rect(0.031, 1, 0.031, 1), color)


class Lava(WorldObj):
    def __init__(self):
        super().__init__('lava', 'red')

    def can_overlap(self):
        return True

    def render(self, img):
        c = (255, 128, 0)

        # Background color
        fill_coords(img, point_in_rect(0, 1, 0, 1), c)

        # Little waves
        for i in range(3):
            ylo = 0.3 + 0.2 * i
            yhi = 0.4 + 0.2 * i
            fill_coords(img, point_in_line(0.1, ylo, 0.3, yhi, r=0.03), (0, 0, 0))
            fill_coords(img, point_in_line(0.3, yhi, 0.5, ylo, r=0.03), (0, 0, 0))
            fill_coords(img, point_in_line(0.5, ylo, 0.7, yhi, r=0.03), (0, 0, 0))
            fill_coords(img, point_in_line(0.7, yhi, 0.9, ylo, r=0.03), (0, 0, 0))


class Wall(WorldObj):
    def __init__(self, color='grey'):
        super().__init__('wall', color)

    def see_behind(self):
        return False

    def render(self, img):
        fill_coords(img, point_in_rect(0, 1, 0, 1), COLORS[self.color])


class Door(WorldObj):
    def __init__(self, color, is_open=False, is_locked=False):
        super().__init__('door', color)
        self.is_open = is_open
        self.is_locked = is_locked

    def can_overlap(self):
        """The agent can only walk over this cell when the door is open"""
        return self.is_open

    def see_behind(self):
        return self.is_open

    def toggle(self, env, pos):
        # If the player has the right key to open the door
        if self.is_locked:
            if isinstance(env.carrying, Key) and env.carrying.color == self.color:
                self.is_locked = False
                self.is_open = True
                return True
            return False

        self.is_open = not self.is_open
        return True

    def encode(self):
        """Encode the a description of this object as a 3-tuple of integers"""

        # State, 0: open, 1: closed, 2: locked
        if self.is_open:
            state = 0
        elif self.is_locked:
            state = 2
        elif not self.is_open:
            state = 1

        return (OBJECT_TO_IDX[self.type] * 20, COLOR_TO_IDX[self.color] * 20, state * 100)

    def render(self, img):
        c = COLORS[self.color]

        if self.is_open:
            fill_coords(img, point_in_rect(0.88, 1.00, 0.00, 1.00), c)
            fill_coords(img, point_in_rect(0.92, 0.96, 0.04, 0.96), (0, 0, 0))
            return

        # Door frame and door
        if self.is_locked:
            fill_coords(img, point_in_rect(0.00, 1.00, 0.00, 1.00), c)
            fill_coords(img, point_in_rect(0.06, 0.94, 0.06, 0.94), 0.45 * np.array(c))

            # Draw key slot
            fill_coords(img, point_in_rect(0.52, 0.75, 0.50, 0.56), c)
        else:
            fill_coords(img, point_in_rect(0.00, 1.00, 0.00, 1.00), c)
            fill_coords(img, point_in_rect(0.04, 0.96, 0.04, 0.96), (0, 0, 0))
            fill_coords(img, point_in_rect(0.08, 0.92, 0.08, 0.92), c)
            fill_coords(img, point_in_rect(0.12, 0.88, 0.12, 0.88), (0, 0, 0))

            # Draw door handle
            fill_coords(img, point_in_circle(cx=0.75, cy=0.50, r=0.08), c)


class Key(WorldObj):
    def __init__(self, color='blue'):
        super(Key, self).__init__('key', color)

    def can_pickup(self):
        return True

    def can_overlap(self):
        return True

    def render(self, img):
        c = COLORS[self.color]

        # Vertical quad
        fill_coords(img, point_in_rect(0.50, 0.63, 0.31, 0.88), c)

        # Teeth
        fill_coords(img, point_in_rect(0.38, 0.50, 0.59, 0.66), c)
        fill_coords(img, point_in_rect(0.38, 0.50, 0.81, 0.88), c)

        # Ring
        fill_coords(img, point_in_circle(cx=0.56, cy=0.28, r=0.190), c)
        fill_coords(img, point_in_circle(cx=0.56, cy=0.28, r=0.064), (0, 0, 0))


class Ball(WorldObj):
    def __init__(self, color='blue'):
        super(Ball, self).__init__('ball', color)

    def can_pickup(self):
        return True

    def render(self, img):
        fill_coords(img, point_in_circle(0.5, 0.5, 0.31), COLORS[self.color])


class Obstacle(WorldObj):
    def __init__(self):
        super(Obstacle, self).__init__('obstacle', 'red')

    def can_pickup(self):
        return True
    
    def see_behind(self):
            return False
    
    def render(self, img):
        c = (255, 128, 0)

        fill_coords(img, point_in_circle(0.5, 0.5, 0.31), c)

        # Little waves
        for i in range(3):
            ylo = 0.3 + 0.2 * i
            yhi = 0.4 + 0.2 * i
            fill_coords(img, point_in_line(0.1, ylo, 0.3, yhi, r=0.03), (0, 0, 0))
            fill_coords(img, point_in_line(0.3, yhi, 0.5, ylo, r=0.03), (0, 0, 0))
            fill_coords(img, point_in_line(0.5, ylo, 0.7, yhi, r=0.03), (0, 0, 0))
            fill_coords(img, point_in_line(0.7, yhi, 0.9, ylo, r=0.03), (0, 0, 0))

class Rubble(WorldObj):
    def __init__(self):
        super().__init__('rubble', 'red')

    def can_pickup(self):
        return True
    
    def see_behind(self):
        return False
    
    def render(self, img):
        c = (255, 128, 0)

        fill_coords(img, point_in_circle(0.5, 0.5, 0.31), c)


class Box(WorldObj):
    def __init__(self, color, contains=None):
        super(Box, self).__init__('box', color)
        self.contains = contains

    def can_pickup(self):
        return True

    def render(self, img):
        c = COLORS[self.color]

        # Outline
        fill_coords(img, point_in_rect(0.12, 0.88, 0.12, 0.88), c)
        fill_coords(img, point_in_rect(0.18, 0.82, 0.18, 0.82), (0, 0, 0))

        # Horizontal slit
        fill_coords(img, point_in_rect(0.16, 0.84, 0.47, 0.53), c)

    def toggle(self, env, pos):
        # Replace the box by its contents
        env.grid.set(*pos, self.contains)
        return True


class Grid:
    """
    Represent a grid and operations on it
    """

    # Static cache of pre-renderer tiles
    tile_cache = {}

    def __init__(self, width, height):
        assert width >= 3
        assert height >= 3

        self.width = width
        self.height = height

        self.grid = [None] * width * height

    def __contains__(self, key):
        if isinstance(key, WorldObj):
            for e in self.grid:
                if e is key:
                    return True
        elif isinstance(key, tuple):
            for e in self.grid:
                if e is None:
                    continue
                if (e.color, e.type) == key:
                    return True
                if key[0] is None and key[1] == e.type:
                    return True
        return False

    def __eq__(self, other):
        grid1 = self.encode()
        grid2 = other.encode()
        return np.array_equal(grid2, grid1)

    def __ne__(self, other):
        return not self == other

    def copy(self):
        from copy import deepcopy
        return deepcopy(self)

    def set(self, i, j, v):
        assert i >= 0 and i < self.width
        assert j >= 0 and j < self.height
        self.grid[j * self.width + i] = v

    def get(self, i, j):
        assert i >= 0 and i < self.width
        assert j >= 0 and j < self.height
        return self.grid[j * self.width + i]

    def horz_wall(self, x, y, length=None, obj_type=Wall):
        if length is None:
            length = self.width - x
        for i in range(0, length):
            self.set(x + i, y, obj_type())

    def vert_wall(self, x, y, length=None, obj_type=Wall):
        if length is None:
            length = self.height - y
        for j in range(0, length):
            self.set(x, y + j, obj_type())

    def wall_rect(self, x, y, w, h):
        self.horz_wall(x, y, w)
        self.horz_wall(x, y+h-1, w)
        self.vert_wall(x, y, h)
        self.vert_wall(x+w-1, y, h)

    def rotate_left(self):
        """
        Rotate the grid to the left (counter-clockwise)
        """

        grid = Grid(self.height, self.width)

        for i in range(self.width):
            for j in range(self.height):
                v = self.get(i, j)
                grid.set(j, grid.height - 1 - i, v)

        return grid

    def slice(self, topX, topY, width, height):
        """
        Get a subset of the grid
        """

        grid = Grid(width, height)

        for j in range(0, height):
            for i in range(0, width):
                x = topX + i
                y = topY + j

                if x >= 0 and x < self.width and \
                   y >= 0 and y < self.height:
                    v = self.get(x, y)
                else:
                    v = Wall()

                grid.set(i, j, v)

        return grid

    @classmethod
    def render_tile(
        cls,
        obj,
        agent_id=None,
        agent_dir=None,
        short_goal_here=False,
        highlight=False,
        tile_size=TILE_PIXELS,
        subdivs=3
    ):
        """
        Render a tile and cache the result
        """
        # Hash map lookup key for the cache

        key = (agent_id, agent_dir, highlight, tile_size, short_goal_here)
        key = obj.encode() + key if obj else key

        if key in cls.tile_cache:
            return cls.tile_cache[key]

        img = np.zeros(shape=(tile_size * subdivs, tile_size * subdivs, 3), dtype=np.uint8)

        # Draw the grid lines (top and left edges)
        fill_coords(img, point_in_rect(0, 0.031, 0, 1), (100, 100, 100))
        fill_coords(img, point_in_rect(0, 1, 0, 0.031), (100, 100, 100))

        if obj != None:
            obj.render(img)

        # Overlay the agent on top
        if agent_dir is not None:

            tri_fn = point_in_triangle(
                (0.12, 0.19),
                (0.87, 0.50),
                (0.12, 0.81),
            )

            # Rotate the agent based on its direction
            tri_fn = rotate_fn(tri_fn, cx=0.5, cy=0.5, theta=0.5*math.pi*agent_dir)
            fill_coords(img, tri_fn, FILL_COLORS[agent_id])

        if short_goal_here:
            tri_fn = point_in_circle(0.5, 0.5, 0.31)
            fill_coords(img, tri_fn, FILL_COLORS[agent_id])

        # Highlight the cell if needed
        if highlight:
            highlight_img(img)

        # Downsample the image to perform supersampling/anti-aliasing
        img = downsample(img, subdivs)

        # Cache the rendered tile
        cls.tile_cache[key] = img

        return img

    def render(
        self,
        num_agents,
        tile_size,
        short_goal_pos=None,
        agent_pos=None,
        agent_dir=None,
        highlight_mask=None
    ):
        """
        Render this grid at a given scale
        :param r: target renderer object
        :param tile_size: tile size in pixels
        """

        if highlight_mask is None:
            highlight_mask = np.zeros(shape=(num_agents, self.width, self.height), dtype=bool)

        # Compute the total grid size
        width_px = self.width * tile_size
        height_px = self.height * tile_size

        img = np.zeros(shape=(height_px, width_px, 3), dtype=np.uint8)

        # Render the grid
        for j in range(0, self.height):
            for i in range(0, self.width):
                cell = self.get(i, j)

                for agent_id in range(num_agents):
                    agent_here = np.array_equal(agent_pos[agent_id], (i, j))
                    a = short_goal_pos[agent_id][0]
                    b = short_goal_pos[agent_id][1]
                    short_goal_here = np.array_equal((int(a), int(b)), (i, j))

                    tile_img = Grid.render_tile(
                        cell,
                        agent_id=agent_id if (agent_here | short_goal_here) else None,
                        agent_dir=agent_dir[agent_id] if agent_here else None,
                        short_goal_here=short_goal_here,
                        highlight=highlight_mask[i, j],
                        tile_size=tile_size
                    )

                    if agent_here | short_goal_here:
                        tile_img = tile_img.copy()
                        for other_agent_id in range(agent_id+1, num_agents):
                            other_agent_here = np.array_equal(agent_pos[other_agent_id], (i, j))
                            a = short_goal_pos[other_agent_id][0]
                            b = short_goal_pos[other_agent_id][1]

                            other_short_goal_here = np.array_equal((int(a), int(b)), (i, j))
                            if other_agent_here and agent_dir[other_agent_id] != agent_dir[agent_id]:
                                other_tile_img = Grid.render_tile(
                                    cell,
                                    agent_id=other_agent_id if (
                                        other_agent_here | other_short_goal_here) else None,
                                    agent_dir=agent_dir[other_agent_id],
                                    short_goal_here=other_short_goal_here,
                                    highlight=highlight_mask[i, j],
                                    tile_size=tile_size
                                )
                                tile_img += other_tile_img
                                tile_img[tile_img > 255] = 255
                        break

                ymin = j * tile_size
                ymax = (j+1) * tile_size
                xmin = i * tile_size
                xmax = (i+1) * tile_size
                img[ymin:ymax, xmin:xmax, :] = tile_img.copy()

        return img

    def render_single(
        self,
        tile_size,
        agent_id=None,
        agent_pos=None,
        agent_dir=None,
        highlight_mask=None
    ):
        """
        Render this grid at a given scale
        :param r: target renderer object
        :param tile_size: tile size in pixels
        """

        if highlight_mask is None:
            highlight_mask = np.zeros(shape=(self.width, self.height), dtype=bool)

        # Compute the total grid size
        width_px = self.width * tile_size
        height_px = self.height * tile_size

        img = np.zeros(shape=(height_px, width_px, 3), dtype=np.uint8)

        # Render the grid
        for j in range(0, self.height):
            for i in range(0, self.width):
                cell = self.get(i, j)

                agent_here = np.array_equal(agent_pos, (i, j))
                tile_img = Grid.render_tile(
                    cell,
                    agent_id=agent_id if agent_here else None,
                    agent_dir=agent_dir if agent_here else None,
                    highlight=highlight_mask[i, j],
                    tile_size=tile_size
                )

                ymin = j * tile_size
                ymax = (j+1) * tile_size
                xmin = i * tile_size
                xmax = (i+1) * tile_size
                img[ymin:ymax, xmin:xmax, :] = tile_img

        return img

    def encode(self, vis_mask=None):
        """
        Produce a compact numpy encoding of the grid
        """

        if vis_mask is None:
            vis_mask = np.ones((self.width, self.height), dtype=bool)

        array = np.zeros((self.width, self.height, 3), dtype='uint8')

        for i in range(self.width):
            for j in range(self.height):
                if vis_mask[i, j]:
                    v = self.get(i, j)

                    if v is None:
                        array[i, j, 0] = OBJECT_TO_IDX['empty'] * 20
                        array[i, j, 1] = 0
                        array[i, j, 2] = 0

                    else:
                        array[i, j, :] = v.encode()

        return array

    @staticmethod
    def decode(array):
        """
        Decode an array grid encoding back into a grid
        """

        width, height, channels = array.shape
        assert channels == 3

        vis_mask = np.ones(shape=(width, height), dtype=bool)

        grid = Grid(width, height)
        for i in range(width):
            for j in range(height):
                type_idx, color_idx, state = array[i, j]
                v = WorldObj.decode(type_idx, color_idx, state)
                grid.set(i, j, v)
                vis_mask[i, j] = (type_idx != OBJECT_TO_IDX['unseen'])

        return grid, vis_mask

    def process_vis(grid, agent_pos):
        ################# First method is for 360 degrees lidar vision ##################
        mask = np.zeros(shape=(grid.width, grid.height), dtype=bool)
        agent_x = agent_pos[0]
        agent_y = agent_pos[1]
        mask[agent_pos[0], agent_pos[1]] = True

        # local_map = grid.encode()[:, :, 0] #saves the object types
        
        visible_cells = []
        for j in range(grid.height):
            cells_on_line = bresenham_line(agent_x, agent_y, 0, j) # left column
            visible_cells += check_visibility(cells_on_line, grid)
            cells_on_line = bresenham_line(agent_x, agent_y, grid.width-1, j) # right column
            visible_cells += check_visibility(cells_on_line, grid)
        # ic(visible_cells)
        for i in range(grid.width):
            cells_on_line = bresenham_line(agent_x, agent_y, i, 0) # top row
            visible_cells += check_visibility(cells_on_line, grid)
            cells_on_line = bresenham_line(agent_x, agent_y, i, grid.height-1) # bottom row
            visible_cells += check_visibility(cells_on_line, grid)
            
        for cell in visible_cells:
            mask[cell[0], cell[1]] = True
        
        return mask
        
        # ##################### second method is for camera vision from yang-xy20/async_mappo #####################
        # mask = np.ones(shape=(grid.width, grid.height), dtype=bool)

        # mask[agent_pos[0], agent_pos[1]] = True

        # local_map = grid.encode()[:, :, 0]

        # for j in range(grid.height):
        #     for i in range(agent_pos[0]-1, -1, -1):
        #         if local_map[i, j] != 20 and local_map[i, j] != 60:
        #             if j == grid.height-1:
        #                 mask[:i+1, :j+1] = False
        #             else:
        #                 mask[:i+1, :j] = False
        #             for h in range(j+1):
        #                 if local_map[i, h] != 0 and (local_map[i+1:agent_pos[0]+1, j].all() == 20 or local_map[i+1:agent_pos[0]+1, j].all() == 60):
        #                     mask[i, h] = True
        #             break

        # for j in range(grid.height):
        #     for i in range(agent_pos[0]+1, grid.width):
        #         if local_map[i, j] != 20 and local_map[i, j] != 60:
        #             if j == grid.height-1:
        #                 mask[i:, :j+1] = False
        #             else:
        #                 mask[i:, :j] = False
        #             for h in range(j+1):
        #                 if local_map[i, h] != 0 and (local_map[agent_pos[0]:i, j].all() == 20 or local_map[agent_pos[0]:i, j].all() == 60):
        #                     mask[i, h] = True
        #             break

        # for j in range(agent_pos[1]-1, -1, -1):
        #     if local_map[agent_pos[0], j] != 20 and local_map[agent_pos[0], j] != 60:
        #         mask[agent_pos[0], :j] = False
        #         break

        # return mask
    ###################### Method on Farama Minigrid ######################
        # mask = np.zeros(shape=(grid.width, grid.height), dtype=bool)

        # mask[agent_pos[0], agent_pos[1]] = True

        # for j in reversed(range(0, grid.height)):
        #     for i in range(0, grid.width - 1):
        #         if not mask[i, j]:
        #             continue

        #         cell = grid.get(i, j)
        #         if cell and not cell.see_behind():
        #             continue

        #         mask[i + 1, j] = True
        #         if j > 0:
        #             mask[i + 1, j - 1] = True
        #             mask[i, j - 1] = True

        #     for i in reversed(range(1, grid.width)):
        #         if not mask[i, j]:
        #             continue

        #         cell = grid.get(i, j)
        #         if cell and not cell.see_behind():
        #             continue

        #         mask[i - 1, j] = True
        #         if j > 0:
        #             mask[i - 1, j - 1] = True
        #             mask[i, j - 1] = True

        # for j in range(0, grid.height):
        #     for i in range(0, grid.width):
        #         if not mask[i, j]:
        #             grid.set(i, j, None)

        # return mask

class MiniGridEnv(gym.Env):
    """
    2D grid world game environment
    """

    metadata = {
        'render.modes': ['searchandrescue', 'rgb_array'],
        'video.frames_per_second': 10
    }

    # Enumeration of possible actions
    class Actions(IntEnum):
        # Turn left, turn right, move forward
        left = 0
        right = 1
        forward = 2
        stop = 3
        infeasible_action = 4
        
        # interact = 4

        # # Pick up an object
        # pickup = 3
        # # Drop an object
        # drop = 4
        # # Toggle/activate an object
        # toggle = 5

        # # Done completing task
        # done = 6

    def __init__(
        self,
        num_agents=2,
        grid_size=None,
        width=None,
        height=None,
        max_steps=100,
        see_through_walls=False,
        seed=1337,
        agent_view_size=7,
        use_full_comm=False,
        use_partial_comm=False,
        use_orientation=False,
        use_stack = True,
        algorithm_name = 'amat'
    ):
        self.num_agents = num_agents
        # Can't set both grid_size and width/height
        if grid_size:
            assert width == None and height == None
            width = grid_size
            height = grid_size

        # Action enumeration for this environment
        self.actions = MiniGridEnv.Actions
        # Actions are discrete integer values
        self.action_space = [spaces.Discrete(len(self.actions)) for _ in range(self.num_agents)] #this is changed in GridWorld_Env

        # Number of cells (width and height) in the agent view
        assert agent_view_size % 2 == 1
        assert agent_view_size >= 3
        self.agent_view_size = agent_view_size
        self.full_w = grid_size  + 2*self.agent_view_size
        self.full_h = grid_size  + 2*self.agent_view_size
        self.width = width
        self.height = height
        
        # Observation space is set later in GridWorld_Env
        self.observation_space = []
        self.share_observation_space = []

        # Range of possible rewards
        self.reward_range = (0, 1)

        # Window to use for human rendering mode
        self.window = None

        # Environment configuration
        self.width = width
        self.height = height
        self.max_steps = max_steps
        self.see_through_walls = see_through_walls

    def reset(self, choose=True):
        # Current position and direction of the agent
        self.agent_pos = []
        self.agent_dir = []
        self.no_wall_size = 0
        self.prev_dist = np.zeros((self.num_agents,1)) # to be used in the reward function
        self.agents_time_until_action = np.full((self.num_agents,1), 0) # Time until the agent can take an action, used to set different speeds for different agents
        self.rubble_cells_reached = set()
        self.rubble_cells_attended = []

        # self.target_reached = 0

        # Generate a new random grid at the start of each episode
        # To keep the same grid for each episode, call env.seed() with
        # the same seed before calling env.reset()
        self.overall_gen_grid(self.width, self.height)


        for i in range(self.width):
            for j in range(self.height):
                c = self.grid.get(i, j)
                if c == None:
                    self.no_wall_size += 1
                elif c.type != 'wall' and c.type != 'obstacle':
                    self.no_wall_size += 1
        # These fields should be defined by _gen_grid
        assert len(self.agent_pos) != 0
        assert len(self.agent_dir) != 0

        for agent_id in range(self.num_agents):
            # Set different speeds for different agents
            if self.agent_types_list[agent_id] == 0: # Actuator agent
                self.agents_time_until_action[agent_id] = 1
            elif self.agent_types_list[agent_id] == 2: # Engineer agent
                self.agents_time_until_action[agent_id] = 1
            
            self.rubble_cells_attended.append(set())

            # Check that the agent doesn't overlap with an object
            start_cell = self.grid.get(*self.agent_pos[agent_id])
            assert start_cell is None or start_cell.can_overlap()

        # # Item picked up, being carried, initially nothing
        # self.carrying = {}
        # for agent_id in range(self.num_agents):
        #     self.carrying[agent_id] = None
        # self.agent_inventory = np.zeros((self.num_agents,2))
        # self.agent_inventory[:,0] = 1

        # Step count since episode start
        self.step_count = 0

        # Return first observation
        obs = [self.gen_obs(agent_id) for agent_id in range(self.num_agents)]
        return obs

    def seed(self, seed=1337):
        # Seed the random number generator
        self.np_random, _ = seeding.np_random(seed)
        return [seed]

    def hash(self, agent_id, size=16):
        """Compute a hash that uniquely identifies the current state of the environment.
        :param size: Size of the hashing
        """
        sample_hash = hashlib.sha256()

        to_encode = [self.grid.encode(), self.agent_pos[agent_id], self.agent_dir[agent_id]]
        for item in to_encode:
            sample_hash.update(str(item).encode('utf8'))

        return sample_hash.hexdigest()[:size]

    @property
    def steps_remaining(self):
        return self.max_steps - self.step_count

    def __str__(self):
        """
        Produce a pretty string of the environment's grid along with the agent.
        A grid cell is represented by 2-character string, the first one for
        the object and the second one for the color.
        """

        # Map of object types to short string
        OBJECT_TO_STR = {
            'wall': 'W',
            'floor': 'F',
            'door': 'D',
            'key': 'K',
            'ball': 'A',
            'box': 'B',
            'obstacle': 'O',
            'target': 'G',
            'lava': 'V',
            'rubble': 'R',
            'trace': 'T',
        }

        # Short string for opened door
        OPENDED_DOOR_IDS = '_'

        # Map agent's direction to short string
        AGENT_DIR_TO_STR = {
            0: '>',
            1: 'V',
            2: '<',
            3: '^'
        }

        str = ''

        for j in range(self.grid.height):
            for i in range(self.grid.width):

                agent_is_here = False
                for agent_id in range(self.num_agents):
                    if i == self.agent_pos[agent_id][0] and j == self.agent_pos[agent_id][1]:
                        str += 2 * AGENT_DIR_TO_STR[self.agent_dir[agent_id]]
                        agent_is_here = True

                if agent_is_here:
                    continue

                c = self.grid.get(i, j)

                if c == None:
                    str += '  '
                    continue

                if c.type == 'door':
                    if c.is_open:
                        str += '__'
                    elif c.is_locked:
                        str += 'L' + c.color[0].upper()
                    else:
                        str += 'D' + c.color[0].upper()
                    continue

                str += OBJECT_TO_STR[c.type] + c.color[0].upper()

            if j < self.grid.height - 1:
                str += '\n'

        return str

    def _gen_grid(self, width, height):
        assert False, "_gen_grid needs to be implemented by each environment"

    def _reward(self):
        """
        Compute the reward to be given upon success
        """

        # return 1 - 0.75*(self.step_count / self.max_steps)
        return 1 - 0.9*(self.step_count / self.max_steps)

    def _l1_distance_ratio(self, x, y):
        """
        Compute the ratio of distance between two objects and the maximum distance
        """

        return (l1distance(x,y))/l1distance((0,0),(self.width-1, self.height-1))
    
    def _euc_distance_ratio(self, x, y):
        """
        Compute the ratio of distance between two objects and the maximum distance
        """

        return (euclideandistance(x,y))/euclideandistance((0,0),(self.width-1, self.height-1))
    def _penalty(self):
        return -1.0

    def _rand_int(self, low, high):
        """
        Generate random integer in [low,high[
        """

        return self.np_random.integers(low, high)

    def _rand_float(self, low, high):
        """
        Generate random float in [low,high[
        """

        return self.np_random.uniform(low, high)

    def _rand_bool(self):
        """
        Generate random boolean value
        """

        return (self.np_random.integers(0, 2) == 0)

    def _rand_elem(self, iterable):
        """
        Pick a random element in a list
        """

        lst = list(iterable)
        idx = self._rand_int(0, len(lst))
        return lst[idx]

    def _rand_subset(self, iterable, num_elems):
        """
        Sample a random subset of distinct elements of a list
        """

        lst = list(iterable)
        assert num_elems <= len(lst)

        out = []

        while len(out) < num_elems:
            elem = self._rand_elem(lst)
            lst.remove(elem)
            out.append(elem)

        return out

    def _rand_color(self):
        """
        Generate a random color name (string)
        """

        return self._rand_elem(COLOR_NAMES)

    def _rand_pos(self, xLow, xHigh, yLow, yHigh):
        """
        Generate a random (x,y) position tuple
        """

        return (
            self.np_random.randint(xLow, xHigh),
            self.np_random.randint(yLow, yHigh)
        )

    def place_obj(self,
                  obj,
                  top=None,
                  size=None,
                  reject_fn=None,
                  max_tries=1000
                  ):
        """
        Place an object at an empty position in the grid
        :param top: top-left position of the rectangle where to place
        :param size: size of the rectangle where to place
        :param reject_fn: function to filter out potential positions
        """

        if top is None:
            top = (0, 0)
        else:
            top = (max(top[0], 0), max(top[1], 0))

        if size is None:
            size = (self.grid.width, self.grid.height)

        num_tries = 0

        while True:
            # This is to handle with rare cases where rejection sampling
            # gets stuck in an infinite loop
            if num_tries > max_tries:
                # print("Failed to place object")
                return np.array((-1, -1))
                # raise RecursionError('rejection sampling failed in place_obj')

            num_tries += 1

            pos = np.array((
                self._rand_int(top[0], min(top[0] + size[0], self.grid.width)),
                self._rand_int(top[1], min(top[1] + size[1], self.grid.height))
            ))

            # Don't place the object on top of another object
            if self.grid.get(*pos) != None:
                continue

            # Don't place the object where the agent is
            agent_is_here = False
            for agent_pos in self.agent_pos:
                if np.array_equal(pos, agent_pos):
                    agent_is_here = True

            if agent_is_here:
                continue

            # Check if there is a filtering criterion
            if reject_fn and reject_fn(self, pos):
                continue

            break

        self.grid.set(*pos, obj)

        if obj is not None:
            obj.init_pos = pos
            obj.cur_pos = pos

        return pos

    def put_obj(self, obj, i, j):
        """
        Put an object at a specific position in the grid
        """

        self.grid.set(i, j, obj)
        obj.init_pos = (i, j)
        obj.cur_pos = (i, j)

    def place_agent(
        self,
        top=None,
        size=None,
        rand_dir=True,
        max_tries=math.inf,
        use_same_location=False,
    ):
        """
        Set the agent's starting point at an empty position in the grid
        """

        self.agent_pos = []
        self.agent_dir = []
        pos = []
        if use_same_location:
            p = self.place_obj(None, top, size, max_tries=max_tries)
            for agent_id in range(self.num_agents):
                self.agent_pos.append(p)
                pos.append(p)
                if rand_dir:
                    self.agent_dir.append(self._rand_int(0, 4))
        else:
            for agent_id in range(self.num_agents):
                p = self.place_obj(None, top, size, max_tries=max_tries)
                self.agent_pos.append(p)
                pos.append(p)
                if rand_dir:
                    self.agent_dir.append(self._rand_int(0, 4))
        return pos

    def dir_vec(self, agent_id):
        """
        Get the direction vector for the agent, pointing in the direction
        of forward movement.
        """

        assert self.agent_dir[agent_id] >= 0 and self.agent_dir[agent_id] < 4
        return DIR_TO_VEC[self.agent_dir[agent_id]]

    def right_vec(self, agent_id):
        """
        Get the vector pointing to the right of the agent.
        """

        dx, dy = self.dir_vec(agent_id)
        return np.array((-dy, dx))

    def front_pos(self, agent_id):
        """
        Get the position of the cell that is right in front of the agent
        """

        return self.agent_pos[agent_id] + self.dir_vec(agent_id)

    def get_view_coords(self, agent_id, i, j):
        """
        Translate and rotate absolute grid coordinates (i, j) into the
        agent's partially observable view (sub-grid). Note that the resulting
        coordinates may be negative or outside of the agent's view size.
        """
        # if self.agent_types_list[agent_id] == 0: # actuator agents have a smaller view size (needs to be modified to work properly)
        #     agent_view_size = self.agent_view_size
        # else:
        agent_view_size = self.agent_view_size


        ax, ay = self.agent_pos[agent_id]
        dx, dy = self.dir_vec(agent_id)
        rx, ry = self.right_vec(agent_id)

        # Compute the absolute coordinates of the top-left view corner
        sz = agent_view_size
        hs = agent_view_size // 2
        tx = ax + (dx * (sz-1)) - (rx * hs)
        ty = ay + (dy * (sz-1)) - (ry * hs)

        lx = i - tx
        ly = j - ty

        # Project the coordinates of the object relative to the top-left
        # corner onto the agent's own coordinate system
        vx = (rx*lx + ry*ly)
        vy = -(dx*lx + dy*ly)

        return vx, vy

    def get_view_exts(self, agent_id):
        """
        Get the extents of the square set of tiles visible to the agent
        Note: the bottom extent indices are not included in the set
        """
        agent_pos = self.agent_pos[agent_id]
        agent_dir = self.agent_dir[agent_id]
    
        # if self.agent_types_list[agent_id] == 0: # actuator agents have a smaller view size
        #     agent_view_size = self.agent_view_size 
        # else:
        agent_view_size = self.agent_view_size

        ### This is front camera view
        # # Facing right
        # if agent_dir == 0:
        #     topX = agent_pos[0]
        #     topY = agent_pos[1] - agent_view_size // 2
        # # Facing down
        # elif agent_dir == 1:
        #     topX = agent_pos[0] - agent_view_size // 2
        #     topY = agent_pos[1]
        # # Facing left
        # elif agent_dir == 2:
        #     topX = agent_pos[0] - agent_view_size + 1
        #     topY = agent_pos[1] - agent_view_size // 2
        # # Facing up
        # elif agent_dir == 3:
        #     topX = agent_pos[0] - agent_view_size // 2
        #     topY = agent_pos[1] - agent_view_size + 1
        # else:
        #     assert False, "invalid agent direction"

        # botX = topX + agent_view_size
        # botY = topY + agent_view_size

        ### This is 360 degrees camera view
        # Facing right
        topX = agent_pos[0] - agent_view_size // 2
        topY = agent_pos[1] - agent_view_size // 2

        botX = topX + agent_view_size
        botY = topY + agent_view_size

        return (topX, topY, botX, botY)

    def relative_coords(self, agent_id, x, y):
        """
        Check if a grid position belongs to the agent's field of view, and returns the corresponding coordinates
        """
        # if self.agent_types_list[agent_id] == 0: # actuator agents have a smaller view size
        #     agent_view_size = self.agent_view_size
        # else:
        agent_view_size = self.agent_view_size
        vx, vy = self.get_view_coords(agent_id, x, y)

        if vx < 0 or vy < 0 or vx >= agent_view_size or vy >= agent_view_size:
            return None

        return vx, vy

    def in_view(self, agent_id, x, y):
        """
        check if a grid position is visible to the agent
        """

        return self.relative_coords(agent_id, x, y) is not None

    def agent_sees(self, agent_id, x, y):
        """
        Check if a non-empty grid position is visible to the agent
        """

        coordinates = self.relative_coords(agent_id, x, y)
        if coordinates is None:
            return False
        vx, vy = coordinates

        obs = self.gen_obs(agent_id)
        obs_grid, _ = Grid.decode(obs['image'])
        obs_cell = obs_grid.get(vx, vy)
        world_cell = self.grid.get(x, y)

        return obs_cell is not None and obs_cell.type == world_cell.type
    
    def adjacent_cells(self, x, y, surround=False):
        """
        Get the list of cells adjacent to a given cell
        """
        adj = []
        if surround:
            steps = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        else:
            steps = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for dx, dy in steps:
            if x+dx >= 0 and x+dx < self.width and y+dy >= 0 and y+dy < self.height:
                adj.append((x+dx, y+dy))
        

        return adj

    def step(self, action):

        self.step_count += 1
        
        reward = np.zeros((self.num_agents,1))
        dist = np.zeros((self.num_agents,1)) #distance to target
        done = False
        obs = []

        # rubble_adjacent_cells = []
        # for rubble_pos in self.all_rubble_set:
        #     rubble_adjacent_cells.append(self.adjacent_cells(*rubble_pos))
        

        # Penalty for not having found the target and reward for finding it (removed for now)

        for agent_id in range(self.num_agents):
            # Check if the agent can take an action
            if (action[agent_id] != self.actions.stop) and (action[agent_id] != self.actions.infeasible_action):
                if self.agents_time_until_action[agent_id] > 0:
                    self.agents_time_until_action[agent_id] -= 1
                    continue

            
            # Get the position in front of the agent
            fwd_pos = self.front_pos(agent_id)

            # Get the contents of the cell in front of the agent
            fwd_cell = self.grid.get(*fwd_pos)
            
            # print("agent {} action is {}".format(agent_id, action[agent_id]))
            # print("fwd cell type is {}".format(fwd_cell.type))

            
            if action[agent_id] == -2: # unavailable action
                # reward[agent_id] += 5*self._penalty()
                pass
            
            # Rotate left
            elif action[agent_id] == self.actions.left:
                # print("left")
                self.agent_dir[agent_id] -= 1
                if self.agent_dir[agent_id] < 0:
                    self.agent_dir[agent_id] += 4
                if self.agent_types_list[agent_id] == 0:
                    self.agents_time_until_action[agent_id] = 1
                elif self.agent_types_list[agent_id] == 2:
                    self.agents_time_until_action[agent_id] = 1
            
            # Rotate right
            elif action[agent_id] == self.actions.right:
                # print("right")
                self.agent_dir[agent_id] = (self.agent_dir[agent_id] + 1) % 4
                if self.agent_types_list[agent_id] == 0:
                    self.agents_time_until_action[agent_id] = 1
                elif self.agent_types_list[agent_id] == 2:
                    self.agents_time_until_action[agent_id] = 1
            
               
            # Move forward
            elif action[agent_id] == self.actions.forward:
                if fwd_cell == None or fwd_cell.can_overlap():
                    can_pass = True
                    for j in range(self.num_agents): # use when agents can't pass through each other (needs a fix: when two agents face each other, they might get stuck)
                        if j != agent_id:
                            if tuple(fwd_pos) == tuple(self.agent_pos[j]):
                                can_pass = False
                                break
                    if can_pass:
                        self.agent_pos[agent_id] = fwd_pos
                        if self.use_energy_penalty:
                            reward[agent_id] += 0.05*self._penalty() #energy cost of moving
                        if self.agent_types_list[agent_id] == 0:
                            self.agents_time_until_action[agent_id] = 1
                        elif self.agent_types_list[agent_id] == 2:
                            self.agents_time_until_action[agent_id] = 1
                            # if self.carrying[agent_id]:
                            #     reward[agent_id] += 5*self._penalty() #energy cost of carrying rubble

                if fwd_cell != None and fwd_cell.type == 'target' and self.agent_types_list[agent_id]==0:
                    
                    pass
                if fwd_cell != None and fwd_cell.type == 'obstacle':
                    # pass
                    reward[agent_id] += self._penalty()
                if fwd_cell != None and fwd_cell.type == 'rubble':
                    # pass
                    reward[agent_id] += self._penalty()
                if fwd_cell != None and fwd_cell.type == 'lava':
                    pass
                if fwd_cell != None and fwd_cell.type == 'wall':
                    # pass
                    reward[agent_id] += self._penalty()
                
               
            # Stop
            elif action[agent_id] == self.actions.stop:

                if self.agent_types_list[agent_id] == 0 and self.target_found[agent_id]: #for actuator agents
                        l1dist_to_target = l1distance(self.agent_pos[agent_id], self.target_pos) #TODO: add support for multiple targets
                        if l1dist_to_target <=1 and self.target_rescued == 0:
                            reward += 300*self._reward() # Team's success reward
                            self.target_rescued = 1
                        else:
                            pass
                elif self.agent_types_list[agent_id] == 2:
                    if len(self.all_rubble_set) != 0:
                        eng_pos = self.agent_pos[agent_id]
                        adjacent_to_eng = self.adjacent_cells(*eng_pos)
                        for cell_pos in adjacent_to_eng:
                            if self.grid.get(*cell_pos) != None and self.grid.get(*cell_pos).type == 'rubble' and tuple(cell_pos) not in self.rubble_cells_reached:
                                reward[agent_id] += 20*self._reward()
                                self.rubble_cells_reached.add(tuple(cell_pos))
                                self.all_rubble_set.discard(tuple(cell_pos))
                                self.agent_rubble_sets[agent_id].discard(tuple(cell_pos))
                                self.grid.set(*cell_pos, None)
                                break

            elif action[agent_id] == self.actions.infeasible_action:
                # reward[agent_id] += self._penalty()
                pass

            # Pick up an object
            elif action[agent_id] == self.actions.pickup:
                if fwd_cell and fwd_cell.can_pickup():
                    if self.carrying is None:
                        self.carrying = fwd_cell
                        self.carrying.cur_pos = np.array([-1, -1])
                        self.grid.set(*fwd_pos, None)

            # Drop an object
            elif action[agent_id] == self.actions.drop:
                if not fwd_cell and self.carrying:
                    self.grid.set(*fwd_pos, self.carrying)
                    self.carrying.cur_pos = fwd_pos
                    self.carrying = None

            # Toggle/activate an object
            elif action[agent_id] == self.actions.toggle:
                if fwd_cell:
                    fwd_cell.toggle(self, fwd_pos)

            # Done action (not used by default)
            elif action[agent_id] == self.actions.done:
                pass

            else:
                assert False, "unknown action"

            # Penalty for time passing (not used by default)
            if self.use_time_penalty:
                reward[agent_id] += 0.2*self._penalty()            
            
            if self.step_count >= self.max_steps:
                done = True
        
  
        obs = [self.gen_obs(agent_id) for agent_id in range(self.num_agents)]
        # print("rewards at this step are: ", reward)
        return obs, reward, done

    def gen_obs_grid(self, agent_id):
        """
        Generate the sub-grid observed by the agent.
        This method also outputs a visibility mask telling us which grid
        cells the agent can actually see.
        """
        if self.agent_types_list[agent_id] == 0: # actuator agents have a smaller view size
            agent_view_size = self.agent_view_size 
        else:
            agent_view_size = self.agent_view_size

        topX, topY, botX, botY = self.get_view_exts(agent_id)

        grid = self.grid.slice(topX, topY, agent_view_size, agent_view_size)
        
        ### Rotate the grid for front camera view
        # for i in range(self.agent_dir[agent_id] + 1):
        #     grid = grid.rotate_left()

        # Process occluders and visibility
        # Note that this incurs some performance cost
        if not self.see_through_walls:
            vis_mask = grid.process_vis(agent_pos=(agent_view_size // 2, agent_view_size // 2))
        else:
            vis_mask = np.ones(shape=(grid.width, grid.height), dtype=bool)
        # ic(vis_mask)
         
        # Make it so the agent sees what it's carrying
        # We do this by placing the carried object at the agent's position
        # in the agent's partially observable view
        # agent_pos = grid.width // 2, grid.height -1
        # if self.carrying[agent_id]:
        #     grid.set(*agent_pos, self.carrying[agent_id])
        # else:
        #     grid.set(*agent_pos, None)

        return grid, vis_mask

    def gen_obs(self, agent_id):
        """
        Generate the agent's view (partially observable, low-resolution encoding)
        """

        grid, vis_mask = self.gen_obs_grid(agent_id)

        # Encode the partially observable view into a numpy array
        image = grid.encode(vis_mask)
        # ic(image)
        assert hasattr(self, 'mission'), "environments must define a textual mission string"

        # Observations are dictionaries containing:
        # - an image (partially observable view of the environment)
        # - the agent's direction/orientation (acting as a compass)
        # - a textual mission string (instructions for the agent)
        obs = {
            'image': image,
            'direction': self.agent_dir[agent_id],
            'mission': self.mission
        }

        return obs

    def get_obs_render(self, obs, tile_size=TILE_PIXELS//2):
        """
        Render an agent observation for visualization
        """
        img = []
        for agent_id in range(self.num_agents):
            if self.agent_types_list[agent_id] == 0: # actuator agents have a smaller view size
                agent_view_size = self.agent_view_size
            else:
                agent_view_size = self.agent_view_size
            
            grid, vis_mask = Grid.decode(obs[agent_id])

            # Render the whole grid
            img.append(grid.render_single(
                tile_size,
                agent_id,
                agent_pos=(agent_view_size // 2, agent_view_size // 2),
                agent_dir=3,
                highlight_mask=vis_mask
            ))

        return img

    def render(self, mode='human', short_goal_pos=None, close=False, highlight=True, tile_size=TILE_PIXELS, first=False):
        """
        Render the whole-grid human view
        """

        if close:
            if self.window:
                self.window.close()
            return

        if not self.window:
            from hetmarl.envs.gridworld.gym_minigrid.window import Window
            self.window = Window('gym_minigrid')
            self.window.show(block=False)

        # Mask of which cells to highlight
        highlight_mask = np.zeros(shape=(self.width, self.height), dtype=bool)

        for agent_id in range(self.num_agents):
            if self.agent_types_list[agent_id] == 0: # actuator agents have a smaller view size
                agent_view_size = self.agent_view_size 
            else:
                agent_view_size = self.agent_view_size
            # Compute which cells are visible to the agent
            _, vis_mask = self.gen_obs_grid(agent_id)

            ### front camera view
            # # Compute the world coordinates of the bottom-left corner
            # # of the agent's view area 
            # f_vec = self.dir_vec(agent_id)
            # r_vec = self.right_vec(agent_id)
            # top_left = self.agent_pos[agent_id] + f_vec * \
            #     (agent_view_size -1) - r_vec * (agent_view_size // 2)

            # # For each cell in the visibility mask
            # for vis_j in range(0, agent_view_size):
            #     for vis_i in range(0, agent_view_size):
            #         # If this cell is not visible, don't highlight it
            #         if not vis_mask[vis_i, vis_j]:
            #             continue

            #         # Compute the world coordinates of this cell
            #         abs_i, abs_j = top_left - (f_vec * vis_j) + (r_vec * vis_i)

            #         if abs_i < 0 or abs_i >= self.width:
            #             continue
            #         if abs_j < 0 or abs_j >= self.height:
            #             continue

            #         # Mark this cell to be highlighted
            #         highlight_mask[abs_i, abs_j] = True
            
            ### 360 degrees view
            # Compute the world coordinates of the top-left corner of the agent's view area 
            top_left = [self.agent_pos[agent_id][0] - agent_view_size // 2, self.agent_pos[agent_id][1] - agent_view_size // 2]
            # For each cell in the visibility mask
            for vis_j in range(0, agent_view_size):
                for vis_i in range(0, agent_view_size):
                    # If this cell is not visible, don't highlight it
                    if not vis_mask[vis_i, vis_j]:
                        continue

                    # Compute the world coordinates of this cell
                    abs_i = top_left[0] + vis_i
                    abs_j = top_left[1] + vis_j

                    if abs_i < 0 or abs_i >= self.width:
                        continue
                    if abs_j < 0 or abs_j >= self.height:
                        continue

                    # Mark this cell to be highlighted
                    highlight_mask[abs_i, abs_j] = True

        explore_mask = highlight_mask if first else self.explored_map.T

        img = self.grid.render(
            self.num_agents,
            tile_size,
            short_goal_pos,
            self.agent_pos,
            self.agent_dir,
            highlight_mask=explore_mask if highlight else None
        )

        local_img = self.grid.render(
            self.num_agents,
            tile_size,
            short_goal_pos,
            self.agent_pos,
            self.agent_dir,
            highlight_mask=highlight_mask if highlight else None
        )

        self.window.set_caption(self.mission)
        self.window.show_img(img, local_img)

        return img, local_img

    def close(self):
        if self.window:
            self.window.close()
        return
