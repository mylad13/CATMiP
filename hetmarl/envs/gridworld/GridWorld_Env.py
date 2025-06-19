import gym
from hetmarl.envs.gridworld.gym_minigrid.register import register
import numpy as np
from hetmarl.utils.multi_discrete import MultiDiscrete

class GridWorldEnv(object):
    def __init__(self, args):

        self.num_agents = args.num_agents
        self.scenario_name = args.scenario_name
        self.agent_pos = args.agent_pos
        self.num_obstacles = args.num_obstacles
        register(
            id=self.scenario_name,
            grid_size=args.grid_size,
            max_steps=args.max_steps,
            local_step_num=args.local_step_num,
            agent_view_size=args.agent_view_size,
            num_agents=self.num_agents,
            num_obstacles=self.num_obstacles,
            agent_pos=self.agent_pos,
            use_full_comm=args.use_full_comm,
            use_partial_comm=args.use_partial_comm,
            com_sigma=args.com_sigma,
            use_orientation=args.use_orientation,
            use_same_location=args.use_same_location,
            use_agent_obstacle=args.use_agent_obstacle,
            use_time_penalty=args.use_time_penalty,
            use_energy_penalty=args.use_energy_penalty,
            use_intrinsic_reward=args.use_intrinsic_reward,
            use_slam_noise = args.use_slam_noise,
            slam_noise_prob = args.slam_noise_prob,
            entry_point='hetmarl.envs.gridworld.gym_minigrid.envs:SearchAndRescueEnv',
            algorithm_name = args.algorithm_name,
            trajectory_forget_rate=args.trajectory_forget_rate,
            n_agent_types = args.n_agent_types,
            agent_types_list = args.agent_types_list,
            block_doors = args.block_doors,
            block_chance = args.block_chance,
            detect_traces = args.detect_traces,
            action_size = args.action_size,
            )
        self.env = gym.make(self.scenario_name)
        self.max_steps = self.env.max_steps

        # Action space is grid size
        # self.action_space = [
        #     MultiDiscrete([[0, args.grid_size - 1],[0, args.grid_size - 1]])
        #     for _ in range(self.num_agents)]
        
        # Action Space is constant
        action_size = args.action_size
        # self.action_space = [
        #     MultiDiscrete([[-action_size, action_size],[-action_size, action_size]])
        #     for _ in range(self.num_agents)]
        action_size_length = 2*action_size + 1
        self.action_space = [
            gym.spaces.Discrete(action_size_length*action_size_length)
            for _ in range(self.num_agents)
        ]


        global_observation_space = {}
        if args.algorithm_name == 'mat' or args.algorithm_name == 'amat':
            global_observation_space['agent_class_identifier'] = gym.spaces.Box(low=0, high=1, shape=(args.n_agent_types,), dtype='uint8')
            global_observation_space['global_agent_map'] = gym.spaces.Box(low=0, high=1, shape=(7 ,args.grid_size, args.grid_size), dtype='float')
            global_observation_space['local_agent_map'] = gym.spaces.Box(low=0, high=1, shape=(6, 7, 7), dtype='float')
        elif args.algorithm_name == 'mancp':
            global_observation_space['agent_class_identifier'] = gym.spaces.Box(low=0, high=1, shape=(args.n_agent_types,), dtype='uint8')
            global_observation_space['global_agent_map'] = gym.spaces.Box(low=0, high=1, shape=(7 ,args.grid_size, args.grid_size), dtype='float')
            global_observation_space['local_agent_map'] = gym.spaces.Box(low=0, high=1, shape=(6, 7, 7), dtype='float')
            global_observation_space['timespan'] = gym.spaces.Box(low=0, high=30, shape=(1,), dtype='uint8')
        elif args.algorithm_name[:2] == "ft":
            pass
        else:
            raise NotImplementedError

        # share_global_observation_space = global_observation_space.copy()

        global_observation_space = gym.spaces.Dict(global_observation_space)
        # share_global_observation_space = gym.spaces.Dict(share_global_observation_space)

        self.observation_space = []
        self.share_observation_space = []

        for _ in range(self.num_agents):
            self.observation_space.append(global_observation_space)
            # self.share_observation_space.append(share_global_observation_space)


        

    def seed(self, seed=None):
        if seed is None:
            self.env.seed(1)
        else:
            self.env.seed(seed)

    def reset(self, choose=True):
        # self.exploration_reward_weight = 1
        if choose:
            obs, info = self.env.reset()
        else:
            obs = [
                {
                    'image': np.zeros((self.env.width, self.env.height, 3), dtype='uint8'),
                    'direction': 0,
                    'mission': " "
                } for agent_id in range(self.num_agents)
            ]
            info = {}
        return obs, info

    def step(self, actions):
        if not np.all(actions == np.ones((self.num_agents, 1)).astype(np.int32) * (-1.0)):
            obs, rewards, done, infos = self.env.step(actions)
            dones = np.array([done for agent_id in range(self.num_agents)])
        else:
            print("actions are all -1")
            obs = [
                {
                    'image': np.zeros((self.env.width, self.env.height, 3), dtype='uint8'),
                    'direction': 0,
                    'mission': " "
                } for agent_id in range(self.num_agents)
            ]
            rewards = np.zeros((self.num_agents, 1))
            dones = np.array([True for agent_id in range(self.num_agents)])
            infos = {}
        # ic(rewards)
        # ic(dones)
        # ic("about to return stuff to gridworld runner through subprocvecenv")
        return obs, rewards, dones, infos

    def close(self):
        self.env.close()

    def get_short_term_action(self, input):
        outputs = self.env.get_short_term_action(input)
        return outputs

    def render(self, mode="human", short_goal_pos=None):
        if mode == "human":
            self.env.render(mode=mode, short_goal_pos=short_goal_pos)
        else:
            return self.env.render(mode=mode, short_goal_pos=short_goal_pos)
    
    def get_available_actions(self):
        act_dim = self.action_space[0].n
        return self.env.get_available_actions(act_dim)
    
    
    def ft_get_short_term_goals(self, args, mode=""):
        mode_list = ['apf', 'utility', 'nearest', 'rrt', 'voronoi']
        assert mode in mode_list, (f"frontier global mode should be in {mode_list}")
        return self.env.ft_get_short_term_goals(args, mode=mode)

    def ft_get_short_term_actions(self, *args):
        return self.env.ft_get_short_term_actions(*args)
