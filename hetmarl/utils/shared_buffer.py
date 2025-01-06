import torch
import numpy as np
from collections import defaultdict
from icecream import ic

from hetmarl.utils.util import check,get_shape_from_obs_space, get_shape_from_act_space, get_connected_agents

def _flatten(T, N, x):
    return x.reshape(T * N, *x.shape[2:])

def _cast(x):
    return x.transpose(1, 2, 0, 3).reshape(-1, *x.shape[3:])

def _shuffle_agent_grid(x, y):
    rows = np.indices((x, y))[0]
    # cols = np.stack([np.random.permutation(y) for _ in range(x)])
    cols = np.stack([np.arange(y) for _ in range(x)])
    return rows, cols

class SharedReplayBuffer(object):
    """
    Buffer to store training data.
    :param args: (argparse.Namespace) arguments containing relevant model, policy, and env information.
    :param num_agents: (int) number of agents in the env.
    :param obs_space: (gym.Space) observation space of agents.
    :param cent_obs_space: (gym.Space) centralized observation space of agents.
    :param act_space: (gym.Space) action space for agents.
    """
    
    def __init__(self, args, num_agents, obs_space, share_obs_space, act_space, env_name):
        self.episode_length = args.episode_length
        self.num_agents = num_agents
        self.n_rollout_threads = args.n_rollout_threads
        self.hidden_size = args.hidden_size
        self.recurrent_N = args.recurrent_N
        self.recurrent_hidden_size = args.recurrent_hidden_size
        self.gamma = args.gamma
        self.gae_lambda = args.gae_lambda
        self._use_gae = args.use_gae
        self._use_popart = args.use_popart
        self._use_valuenorm = args.use_valuenorm
        self._use_proper_time_limits = args.use_proper_time_limits 
        self.asynch = args.asynch
        self.algo = args.algorithm_name
        self.num_agents = num_agents
        self.env_name = env_name
        self.max_steps = args.max_steps
        self.use_action_masking = args.use_action_masking


        self._mixed_obs = False

        obs_shape = get_shape_from_obs_space(obs_space)
        share_obs_shape = get_shape_from_obs_space(share_obs_space)
        self.obs_shape = obs_shape
        self.share_obs_shape = share_obs_shape

        # for mixed observation
        if 'Dict' in obs_shape.__class__.__name__:
            self._mixed_obs = True
            
            self.obs = {}
            self.share_obs = {}
            self.all_obs = {}

            for key in obs_shape:
                self.obs[key] = np.zeros((self.episode_length + 1, self.n_rollout_threads, num_agents, *obs_shape[key].shape), dtype=np.float32)
                self.all_obs[key] = np.zeros((self.episode_length + 1, self.n_rollout_threads, num_agents, *obs_shape[key].shape), dtype=np.float32) # saves all agent observations everytime an agent acts (asynchronous training)
            for key in share_obs_shape:
                self.share_obs[key] = np.zeros((self.episode_length + 1, self.n_rollout_threads, num_agents, *share_obs_shape[key].shape), dtype=np.float32)
        
        else: 
            # deal with special attn format   
            if type(obs_shape[-1]) == list:
                obs_shape = obs_shape[:1]

            if type(share_obs_shape[-1]) == list:
                share_obs_shape = share_obs_shape[:1]

            self.share_obs = np.zeros((self.episode_length + 1, self.n_rollout_threads, num_agents, *share_obs_shape), dtype=np.float32)
            self.obs = np.zeros((self.episode_length + 1, self.n_rollout_threads, num_agents, *obs_shape), dtype=np.float32)
            self.all_obs = np.zeros((self.episode_length + 1, self.n_rollout_threads, num_agents, *obs_shape), dtype=np.float32) # saves all agent observations everytime an agent acts (asynchronous training)

        self.rnn_states = np.zeros((self.episode_length + 1, self.n_rollout_threads, num_agents, self.recurrent_hidden_size), dtype=np.float32)
        self.rnn_states_critic = np.zeros_like(self.rnn_states)
       
        self.value_preds = np.zeros(
            (self.episode_length + 1, self.n_rollout_threads, num_agents, 1), dtype=np.float32)
        self.returns = np.zeros_like(self.value_preds)
        
        self.advantages = np.zeros(
            (self.episode_length, self.n_rollout_threads, num_agents, 1), dtype=np.float32)
        
        self.act_shape = get_shape_from_act_space(act_space)
        if act_space.__class__.__name__ == 'MultiDiscrete':
            self.act_dim =  act_space.high - act_space.low + 1
            if self.use_action_masking:
                self.available_actions = np.ones((self.episode_length + 1, self.n_rollout_threads, num_agents, *self.act_dim), dtype=np.float32)
            else:
                self.available_actions = None
        elif act_space.__class__.__name__ == 'Discrete':
            self.act_dim = act_space.n
            if self.use_action_masking:
                self.available_actions = np.ones((self.episode_length + 1, self.n_rollout_threads, num_agents, self.act_dim), dtype=np.float32)
            else:
                self.available_actions = None
        self.actions = np.zeros(
            (self.episode_length, self.n_rollout_threads, num_agents, self.act_shape), dtype=np.float32)
        self.action_log_probs = np.full(
            (self.episode_length, self.n_rollout_threads, num_agents, self.act_shape),-1e-9, dtype=np.float32)
        self.rewards = np.zeros(
            (self.episode_length, self.n_rollout_threads, num_agents, 1), dtype=np.float32)

        self.masks = np.zeros((self.episode_length + 1, self.n_rollout_threads, num_agents, 1), dtype=np.float32)
        self.masks[0] = 1
        self.bad_masks = np.ones_like(self.masks)
        self.active_masks = np.ones_like(self.masks) # indicates whether the agent is online or offline (not to be confused with agents that activate to get new actions)

        self.update_step = np.zeros((self.n_rollout_threads, num_agents, 1), dtype=np.int32)
        self.update_step_masks = np.zeros((self.episode_length + 1, self.n_rollout_threads, num_agents, 1), dtype=np.float32)
        self.update_step_masks[0] = 1
        self.active_steps = np.zeros((self.episode_length + 1, self.n_rollout_threads, num_agents, 1), dtype=np.int32)
        self.agent_steps = np.zeros((self.episode_length + 1, self.n_rollout_threads, num_agents, 1), dtype=np.int32)
        self.was_agent_active = np.full((self.episode_length + 1, self.n_rollout_threads, num_agents), False, dtype=bool)
        self.was_agent_active[0] = True

        self.agent_groups = np.ones((self.episode_length + 1, self.n_rollout_threads, num_agents, num_agents), dtype=np.int32)
        self.step = 0

    def insert(self, share_obs, obs, rnn_states, rnn_states_critic, actions, action_log_probs,
               value_preds, rewards, masks, agent_groups, bad_masks=None, active_masks=None, available_actions=None):

        if self._mixed_obs:
            for key in self.share_obs.keys():
                self.share_obs[key][self.step + 1] = share_obs[key].copy()
            for key in self.obs.keys():
                self.obs[key][self.step + 1] = obs[key].copy()
        else:
            self.share_obs[self.step + 1] = share_obs.copy()
            self.obs[self.step + 1] = obs.copy()

        self.rnn_states[self.step + 1] = rnn_states.copy()
        self.rnn_states_critic[self.step + 1] = rnn_states_critic.copy()
        self.actions[self.step] = actions.copy()
        self.action_log_probs[self.step] = action_log_probs.copy()
        self.value_preds[self.step] = value_preds.copy()
        self.rewards[self.step] = rewards.copy()
        self.masks[self.step + 1] = masks.copy()
        if bad_masks is not None:
            self.bad_masks[self.step + 1] = bad_masks.copy()
        if active_masks is not None:
            self.active_masks[self.step + 1] = active_masks.copy()
        if available_actions is not None:
            self.available_actions[self.step + 1] = available_actions.copy()

        self.agent_groups[self.step + 1] = agent_groups.copy()
        
        self.step = (self.step + 1) % self.episode_length
    
    def async_insert(self, share_obs, obs, rnn_states, rnn_states_critic, actions, action_log_probs,
               value_preds, rewards, masks, agent_groups, bad_masks=None, active_masks=None, available_actions=None, active_agents=None):
        assert active_agents is not None
        # for key in self.obs.keys():
        #     self.obs[key][self.step + 1] = obs[key].copy()

        for e, a, agent_step in active_agents:
            agent_step = int(agent_step - 1)
            self.masks[agent_step + 1, e, a] = masks[e, a].copy()

            for key in self.obs.keys():
                self.obs[key][agent_step + 1, e, a] = obs[key][e,a].copy()
                self.all_obs[key][self.step + 1, e] = obs[key][e].copy() #observations from all agents are saved 
           
            self.actions[agent_step, e, a] = actions[e, a].copy() #* self.masks[agent_step, e, a]
            self.action_log_probs[agent_step, e, a] = action_log_probs[e, a].copy() #* self.masks[agent_step, e, a]
            self.value_preds[agent_step, e, a] = value_preds[e, a].copy() #* self.masks[agent_step, e, a]
            self.rewards[agent_step, e, a] = rewards[e, a].copy() #* self.masks[agent_step, e, a]
            
            self.rnn_states[agent_step + 1, e, a] = rnn_states[e, a].copy()
            self.rnn_states_critic[agent_step + 1, e, a] = rnn_states_critic[e, a].copy()

            if bad_masks is not None:
                self.bad_masks[agent_step + 1, e, a] = bad_masks[e, a].copy()
            if active_masks is not None:
                self.active_masks[agent_step + 1, e, a] = active_masks[e, a].copy()
            if available_actions is not None:
                self.available_actions[agent_step + 1, e, a] = available_actions[e, a].copy()
        
            self.active_steps[agent_step + 1 , e, a] = (self.step + 1) * self.masks[agent_step + 1, e, a] # saving each agent's corresponding update step
            self.agent_steps[self.step + 1 , e, a] = (agent_step + 1 ) * self.masks[agent_step + 1, e, a]
            self.was_agent_active[self.step + 1 , e, a] = True * self.masks[agent_step + 1, e, a].astype(bool)
            self.update_step[e, a] = agent_step 
            self.update_step_masks[agent_step , e, a] = 1

        self.agent_groups[self.step + 1] = agent_groups.copy()
        self.step = self.step + 1
        
    def update_mask(self, steps): #only called at the end of episode, before compute
        for e in range(self.n_rollout_threads):
            for a in range(self.num_agents):
                invalid_active_steps = self.active_steps[self.update_step[e,a]+1,e,a]
                if invalid_active_steps != 0:
                    self.was_agent_active[invalid_active_steps, e, a] = False
        self.masks = self.masks * self.update_step_masks
        self.active_masks = self.active_masks * self.masks #same as mask for now (might change when implementing agent loss during training)
        

    def after_update(self):
        # Copying data from the last step to first step is not necessary in our case, since episodes are independent. However in the asynch case,
        # we want to reset the step trackers.
        if self.asynch:
            self.step = 0 #reset the step counter after each update
            self.active_steps = np.zeros((self.episode_length + 1, self.n_rollout_threads, self.num_agents, 1), dtype=np.int32)
            self.agent_steps = np.zeros((self.episode_length + 1, self.n_rollout_threads, self.num_agents, 1), dtype=np.int32)
            self.update_step = np.zeros((self.n_rollout_threads, self.num_agents, 1), dtype=np.int32)
            self.update_step_masks = np.zeros((self.episode_length + 1, self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
            self.update_step_masks[0] = 1
            self.masks = np.zeros((self.episode_length + 1, self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
            self.masks[0] = 1
            self.bad_masks = np.ones_like(self.masks)
            self.active_masks = np.ones_like(self.masks)
            self.was_agent_active = np.full((self.episode_length + 1, self.n_rollout_threads, self.num_agents), False, dtype=bool)
            self.was_agent_active[0] = True
            

            obs_shape = self.obs_shape
            share_obs_shape = self.share_obs_shape
            if self._mixed_obs:               
                self.obs = {}
                self.share_obs = {}
                self.all_obs = {}

                for key in obs_shape:
                    self.obs[key] = np.zeros((self.episode_length + 1, self.n_rollout_threads, self.num_agents, *obs_shape[key].shape), dtype=np.float32)
                    self.all_obs[key] = np.zeros((self.episode_length + 1, self.n_rollout_threads, self.num_agents, *obs_shape[key].shape), dtype=np.float32) 
                for key in share_obs_shape:
                    self.share_obs[key] = np.zeros((self.episode_length + 1, self.n_rollout_threads, self.num_agents, *share_obs_shape[key].shape), dtype=np.float32)
            else: 
                # deal with special attn format   
                if type(obs_shape[-1]) == list:
                    obs_shape = obs_shape[:1]

                if type(share_obs_shape[-1]) == list:
                    share_obs_shape = share_obs_shape[:1]

                self.share_obs = np.zeros((self.episode_length + 1, self.n_rollout_threads, self.num_agents, *share_obs_shape), dtype=np.float32)
                self.obs = np.zeros((self.episode_length + 1, self.n_rollout_threads, self.num_agents, *obs_shape), dtype=np.float32)
                self.all_obs = np.zeros((self.max_steps + 1, self.n_rollout_threads, self.num_agents, *obs_shape), dtype=np.float32) 
            
            self.rnn_states = np.zeros((self.episode_length + 1, self.n_rollout_threads, self.num_agents, self.recurrent_hidden_size), dtype=np.float32)
            self.rnn_states_critic = np.zeros_like(self.rnn_states)
            self.value_preds = np.zeros(
            (self.episode_length + 1, self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
            self.returns = np.zeros_like(self.value_preds)
            
            self.advantages = np.zeros(
                (self.episode_length, self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
            self.agent_groups = np.ones((self.episode_length + 1, self.n_rollout_threads, self.num_agents, self.num_agents), dtype=np.int32)
            if self.use_action_masking:
                if self.act_shape == 2:
                    self.available_actions = np.ones((self.episode_length + 1, self.n_rollout_threads, self.num_agents, *self.act_dim), dtype=np.float32)
                elif self.act_shape == 1:
                    self.available_actions = np.ones((self.episode_length + 1, self.n_rollout_threads, self.num_agents, self.act_dim), dtype=np.float32)
            else:
                self.available_actions = None

            self.actions = np.zeros(
                (self.episode_length, self.n_rollout_threads, self.num_agents, self.act_shape), dtype=np.float32)
            self.action_log_probs = np.full(
                (self.episode_length, self.n_rollout_threads, self.num_agents, self.act_shape),-1e-9, dtype=np.float32)
            self.rewards = np.zeros(
                (self.episode_length, self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        else:
            self.step = 0 #reset the step counter after each update
            self.masks = np.zeros((self.episode_length + 1, self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
            self.masks[0] = 1
            self.bad_masks = np.ones_like(self.masks)
            self.active_masks = np.ones_like(self.masks)

            obs_shape = self.obs_shape
            share_obs_shape = self.share_obs_shape
            if self._mixed_obs:               
                self.obs = {}
                self.share_obs = {}

                for key in obs_shape:
                    self.obs[key] = np.zeros((self.episode_length + 1, self.n_rollout_threads, self.num_agents, *obs_shape[key].shape), dtype=np.float32)
                for key in share_obs_shape:
                    self.share_obs[key] = np.zeros((self.episode_length + 1, self.n_rollout_threads, self.num_agents, *share_obs_shape[key].shape), dtype=np.float32)
            else: 
                # deal with special attn format   
                if type(obs_shape[-1]) == list:
                    obs_shape = obs_shape[:1]

                if type(share_obs_shape[-1]) == list:
                    share_obs_shape = share_obs_shape[:1]

                self.share_obs = np.zeros((self.episode_length + 1, self.n_rollout_threads, self.num_agents, *share_obs_shape), dtype=np.float32)
                self.obs = np.zeros((self.episode_length + 1, self.n_rollout_threads, self.num_agents, *obs_shape), dtype=np.float32)
            
            self.rnn_states = np.zeros((self.episode_length + 1, self.n_rollout_threads, self.num_agents, self.recurrent_hidden_size), dtype=np.float32)
            self.rnn_states_critic = np.zeros_like(self.rnn_states)
            self.value_preds = np.zeros(
            (self.episode_length + 1, self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
            self.returns = np.zeros_like(self.value_preds)
            
            self.advantages = np.zeros(
                (self.episode_length, self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
            self.agent_groups = np.ones((self.episode_length + 1, self.n_rollout_threads, self.num_agents, self.num_agents), dtype=np.int32)
            if self.use_action_masking:
                if self.act_shape == 2:
                    self.available_actions = np.ones((self.episode_length + 1, self.n_rollout_threads, self.num_agents, *self.act_dim), dtype=np.float32)
                elif self.act_shape == 1:
                    self.available_actions = np.ones((self.episode_length + 1, self.n_rollout_threads, self.num_agents, self.act_dim), dtype=np.float32)
            else:
                self.available_actions = None

            self.actions = np.zeros(
                (self.episode_length, self.n_rollout_threads, self.num_agents, self.act_shape), dtype=np.float32)
            self.action_log_probs = np.full(
                (self.episode_length, self.n_rollout_threads, self.num_agents, self.act_shape),-1e-9, dtype=np.float32)
            self.rewards = np.zeros(
                (self.episode_length, self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
    

    def async_compute_returns(self, next_values, value_normalizer=None):
        # This is done for all threads and agents
        for e in range(self.n_rollout_threads):
            for a in range(self.num_agents):
                if self._use_gae:
                    self.value_preds[self.update_step[e,a] + 1, e, a] = next_values[e, a]
                else:
                    self.returns[self.update_step[e,a] + 1, e, a] = next_values[e, a]
        final_step = np.max(self.agent_steps)
        if self._use_gae:
            gae = 0
            for step in reversed(range(final_step + 1)):
                if self._use_popart or self._use_valuenorm:
                    delta = self.rewards[step] + self.gamma * value_normalizer.denormalize(self.value_preds[step + 1]) * self.masks[step + 1] \
                        - value_normalizer.denormalize(self.value_preds[step])
                    delta = delta * self.active_masks[step]
                    gae = delta + self.gamma * self.gae_lambda * self.masks[step + 1] * gae 
                    self.advantages[step] = gae
                    self.returns[step] = (gae + value_normalizer.denormalize(self.value_preds[step])) * self.active_masks[step]
                else:
                    delta = self.rewards[step] + self.gamma * self.value_preds[step + 1] * self.masks[step + 1] - self.value_preds[step] 
                    delta = delta * self.active_masks[step]
                    gae = delta + self.gamma * self.gae_lambda * self.masks[step + 1] * gae 
                    self.advantages[step] = gae
                    self.returns[step] = (gae + self.value_preds[step]) * self.active_masks[step]
        else:
            for step in reversed(range(final_step + 1)):
                self.returns[step] = self.returns[step + 1] * self.gamma * self.masks[step + 1] + self.rewards[step]
    
    
    def compute_returns(self, next_value, value_normalizer=None): #TODO: make sure active_masks are used correctly to mask out info after envs are done
        # This is done for all threads and agents
        # print("all masks are: ", self.masks)
        if self._use_proper_time_limits:
            if self._use_gae:
                self.value_preds[-1] = next_value
                gae = 0
                for step in reversed(range(self.rewards.shape[0])):
                    if self._use_popart or self._use_valuenorm:
                        # step + 1
                        delta = self.rewards[step] + self.gamma * value_normalizer.denormalize(self.value_preds[step + 1]) * self.masks[step + 1]  \
                            - value_normalizer.denormalize(self.value_preds[step])
                        gae = delta + self.gamma * self.gae_lambda * gae * self.masks[step + 1] 
                        gae = gae * self.bad_masks[step + 1]
                        self.advantages[step] = gae
                        self.returns[step] = gae + value_normalizer.denormalize(self.value_preds[step])
                    else:
                        delta = self.rewards[step] + self.gamma * self.value_preds[step + 1] * self.masks[step + 1] - self.value_preds[step] 
                        gae = delta + self.gamma * self.gae_lambda * self.masks[step + 1] * gae 
                        gae = gae * self.bad_masks[step + 1]
                        self.advantages[step] = gae
                        self.returns[step] = gae + self.value_preds[step]
            else:
                self.returns[-1] = next_value
                for step in reversed(range(self.rewards.shape[0])):
                    if self._use_popart or self._use_valuenorm:
                        self.returns[step] = (self.returns[step + 1] * self.gamma * self.masks[step + 1] + self.rewards[step]) * self.bad_masks[step + 1] \
                                            + (1 - self.bad_masks[step + 1]) * value_normalizer.denormalize(self.value_preds[step]) 
                    else:
                        self.returns[step] = (self.returns[step + 1] * self.gamma * self.masks[step + 1] + self.rewards[step]) * self.bad_masks[step + 1] \
                                            + (1 - self.bad_masks[step + 1]) * self.value_preds[step]
        else:
            if self._use_gae:
                self.value_preds[-1] = next_value
                gae = 0
                for step in reversed(range(self.episode_length)):
                    if self._use_popart or self._use_valuenorm:
                        delta = self.rewards[step] + self.gamma * value_normalizer.denormalize(self.value_preds[step + 1]) * self.masks[step + 1] \
                            - value_normalizer.denormalize(self.value_preds[step])
                        gae = delta + self.gamma * self.gae_lambda * self.masks[step + 1] * gae 
                        self.advantages[step] = gae
                        self.returns[step] = (gae + value_normalizer.denormalize(self.value_preds[step])) * self.active_masks[step]
                    else:
                        delta = self.rewards[step] + self.gamma * self.value_preds[step + 1] * self.masks[step + 1] - self.value_preds[step] 
                        gae = delta + self.gamma * self.gae_lambda * self.masks[step + 1] * gae 
                        self.advantages[step] = gae
                        self.returns[step] = (gae + self.value_preds[step]) * self.active_masks[step]
                    # print(self.returns)
            else:
                self.returns[-1] = next_value
                for step in reversed(range(self.episode_length)):
                    self.returns[step] = self.returns[step + 1] * self.gamma * self.masks[step + 1] + self.rewards[step]

    def async_feed_forward_generator_transformer(self, advantages, num_mini_batch=None, mini_batch_size=None):
        """
        Yield training data for MLP policies.
        :param advantages: (np.ndarray) advantage estimates.
        :param num_mini_batch: (int) number of minibatches to split the batch into.
        :param mini_batch_size: (int) number of samples in each minibatch.
        """
        # The goal is to align the asyncrounous data exisitng in the buffer and batch them.


        episode_length = self.episode_length
        n_rollout_threads = self.n_rollout_threads
        num_agents = self.num_agents
        max_steps = np.max(self.active_steps)
        
        batch_size = n_rollout_threads * max_steps #batch size is dynamic based on the number of steps before envs were done
        # print("batch size is: ", batch_size)   

        ordered_actions = np.zeros_like(self.actions)[:max_steps+1]
        ordered_action_log_probs = np.zeros_like(self.action_log_probs)[:max_steps+1]
        ordered_value_preds = np.zeros_like(self.value_preds)[:max_steps+1]
        ordered_masks = np.zeros_like(self.masks)[:max_steps+1]
        ordered_active_masks = np.zeros_like(self.active_masks)[:max_steps+1]
        ordered_advantages = np.zeros_like(advantages)[:max_steps+1]
        ordered_returns = np.zeros_like(self.returns)[:max_steps+1]
        if self.available_actions is not None:
            ordered_available_actions = np.zeros_like(self.available_actions)[:max_steps+1]
        ordered_rnn_states = np.zeros_like(self.rnn_states)[:max_steps+1]
        ordered_rnn_states_critic = np.zeros_like(self.rnn_states_critic)[:max_steps+1]
        
        for step in range(max_steps + 1): # put everything in the same order as all_obs
            agents_active_step_indices = self.agent_steps[step][self.was_agent_active[step]] # a list of indices of agent_steps, for each global (macro) step
            
            if len(agents_active_step_indices) == 0:
                continue
            
            list_of_active_agents = np.transpose(self.was_agent_active[step].nonzero())
            temp_actions = []
            temp_action_log_probs = []
            temp_value_preds = []
            temp_masks = []
            temp_active_masks = []
            temp_advantages = []
            temp_returns = []
            temp_available_actions = []
            temp_rnn_states = []
            temp_rnn_states_critic = []
            for i, index in enumerate(agents_active_step_indices):
                squeezed_actions = self.actions[index].squeeze()
                squeezed_actions = np.expand_dims(squeezed_actions, -1) #for discrete action type
                temp_actions.append(squeezed_actions[list_of_active_agents[i][0], list_of_active_agents[i][1]])
                squeezed_action_log_probs = self.action_log_probs[index].squeeze()
                squeezed_action_log_probs = np.expand_dims(squeezed_action_log_probs, -1) #for discrete action type
                temp_action_log_probs.append(squeezed_action_log_probs[list_of_active_agents[i][0], list_of_active_agents[i][1]])
                squeezed_value_preds = self.value_preds[index].squeeze()
                temp_value_preds.append(squeezed_value_preds[list_of_active_agents[i][0], list_of_active_agents[i][1]])
                squeezed_masks = self.masks[index].squeeze()
                temp_masks.append(squeezed_masks[list_of_active_agents[i][0], list_of_active_agents[i][1]])
                squeezed_active_masks = self.active_masks[index].squeeze()
                temp_active_masks.append(squeezed_active_masks[list_of_active_agents[i][0], list_of_active_agents[i][1]])
                squeezed_advantages = advantages[index].squeeze()
                temp_advantages.append(squeezed_advantages[list_of_active_agents[i][0], list_of_active_agents[i][1]])
                squeezed_returns = self.returns[index].squeeze()
                temp_returns.append(squeezed_returns[list_of_active_agents[i][0], list_of_active_agents[i][1]])
                if self.available_actions is not None:
                    squeezed_available_actions = self.available_actions[index].squeeze()
                    temp_available_actions.append(squeezed_available_actions[list_of_active_agents[i][0], list_of_active_agents[i][1]])
                squeezed_rnn_states = self.rnn_states[index].squeeze()
                temp_rnn_states.append(squeezed_rnn_states[list_of_active_agents[i][0], list_of_active_agents[i][1]])
                squeezed_rnn_states_critic = self.rnn_states_critic[index].squeeze()
                temp_rnn_states_critic.append(squeezed_rnn_states_critic[list_of_active_agents[i][0], list_of_active_agents[i][1]])
                

            ordered_actions[step][self.was_agent_active[step]] = np.array(temp_actions)
            ordered_action_log_probs[step][self.was_agent_active[step]] = np.array(temp_action_log_probs)
            ordered_value_preds[step][self.was_agent_active[step]] = np.expand_dims(np.array(temp_value_preds),-1)
            ordered_masks[step][self.was_agent_active[step]] = np.expand_dims(np.array(temp_masks),-1)
            ordered_active_masks[step][self.was_agent_active[step]] = np.expand_dims(np.array(temp_active_masks),-1)
            ordered_advantages[step][self.was_agent_active[step]] = np.expand_dims(np.array(temp_advantages),-1)
            ordered_returns[step][self.was_agent_active[step]] = np.expand_dims(np.array(temp_returns),-1)
            if self.available_actions is not None:
                ordered_available_actions[step][self.was_agent_active[step]] = np.array(temp_available_actions)
            ordered_rnn_states[step][self.was_agent_active[step]] = np.array(temp_rnn_states)
            ordered_rnn_states_critic[step][self.was_agent_active[step]] = np.array(temp_rnn_states_critic) #TODO: check if this is correct
        
        # merges the first two dimensions (episode_length, n_rollout_threads) and keeps (num_agent, dim)
        # also removes empty data after max_steps
        if self._mixed_obs:
            share_obs = {}
            obs = {}
            for key in self.all_obs.keys():
                obs[key] = self.all_obs[key][:max_steps].reshape(-1, *self.all_obs[key].shape[2:]) 
                share_obs[key] = obs[key].copy()           
        else: #not used
            share_obs = self.share_obs[:max_steps].reshape(-1, *self.share_obs.shape[2:])
            obs = self.all_obs[:max_steps].reshape(-1, *self.obs.shape[2:])
        
        actions = ordered_actions[:max_steps].reshape(-1, *ordered_actions.shape[2:])
        
        action_log_probs = ordered_action_log_probs[:max_steps].reshape(-1, *ordered_action_log_probs.shape[2:])

        value_preds = ordered_value_preds[:max_steps].reshape(-1, *ordered_value_preds.shape[2:])

        masks = ordered_masks[:max_steps].reshape(-1, *ordered_masks.shape[2:])
        
        active_masks = ordered_active_masks[:max_steps].reshape(-1, *ordered_active_masks.shape[2:])
        
        advantages = ordered_advantages[:max_steps].reshape(-1, *advantages.shape[2:])
        
        returns = ordered_returns[:max_steps].reshape(-1, *ordered_returns.shape[2:])

        active_agents = self.was_agent_active[:max_steps].reshape(-1, *self.was_agent_active.shape[2:])

        agent_groups = self.agent_groups[:max_steps].reshape(-1, *self.agent_groups.shape[2:])
        
        if self.available_actions is not None:
            available_actions = ordered_available_actions[:max_steps].reshape(-1, *ordered_available_actions.shape[2:])
        
        rnn_states = ordered_rnn_states[:max_steps].reshape(-1, *ordered_rnn_states.shape[2:])
        rnn_states_critic = ordered_rnn_states_critic[:max_steps].reshape(-1, *ordered_rnn_states_critic.shape[2:])
 
        # throws out experiences (batches of steps*threads) where no agent was active
        pruning_indices = []
        
        for index in range(batch_size):
            if not np.any(active_agents[index]):
                pruning_indices.append(index)
        if self._mixed_obs:
            pruned_obs={}
            pruned_share_obs={}
            for key in obs.keys():
                pruned_obs[key] = np.delete(obs[key], pruning_indices, axis=0)
                pruned_share_obs[key] = np.delete(share_obs[key], pruning_indices, axis=0)
        else:
            pruned_obs = np.delete(obs, pruning_indices, axis=0)
            pruned_share_obs = np.delete(share_obs, pruning_indices, axis=0)
        pruned_actions = np.delete(actions, pruning_indices, axis=0)
        pruned_action_log_probs = np.delete(action_log_probs, pruning_indices, axis=0)
        pruned_value_preds = np.delete(value_preds, pruning_indices, axis=0)
        pruned_masks = np.delete(masks, pruning_indices, axis=0)
        pruned_active_masks = np.delete(active_masks, pruning_indices, axis=0)
        pruned_advantages = np.delete(advantages, pruning_indices, axis=0)
        pruned_returns = np.delete(returns, pruning_indices, axis=0)
        pruned_active_agents = np.delete(active_agents, pruning_indices, axis=0)
        pruned_agent_groups = np.delete(agent_groups, pruning_indices, axis=0)
        if self.available_actions is not None:
            pruned_available_actions = np.delete(available_actions, pruning_indices, axis=0)
        pruned_rnn_states = np.delete(rnn_states, pruning_indices, axis=0)
        pruned_rnn_states_critic = np.delete(rnn_states_critic, pruning_indices, axis=0)

        pruned_batch_size = pruned_returns.shape[0]
        # print("pruned batch size is: ", pruned_batch_size)

        # Reorder everything to put active agents first
        permutations = np.zeros((pruned_batch_size, self.num_agents), dtype=np.int32)
        for i in range(pruned_batch_size):
            true_cnt = 0
            false_cnt = 0
            for a in range(self.num_agents):
                if pruned_active_agents[i, a] == False:
                    permutations[i, self.num_agents - 1 - false_cnt] = a
                    false_cnt += 1
                else:
                    permutations[i, true_cnt] = a
                    true_cnt += 1
        if self._mixed_obs:
            reordered_obs={}
            reordered_share_obs={}
            for key in obs.keys():
                reordered_obs[key] = np.array([row[perm] for row, perm in zip(pruned_obs[key], permutations)])
                reordered_share_obs[key] = np.array([row[perm] for row, perm in zip(pruned_share_obs[key], permutations)])
        else:
            reordered_obs = np.array([row[perm] for row, perm in zip(pruned_obs, permutations)])
            reordered_share_obs = np.array([row[perm] for row, perm in zip(pruned_share_obs, permutations)])
        reordered_actions = np.array([row[perm] for row, perm in zip(pruned_actions, permutations)])
        reordered_action_log_probs = np.array([row[perm] for row, perm in zip(pruned_action_log_probs, permutations)])
        reordered_value_preds = np.array([row[perm] for row, perm in zip(pruned_value_preds, permutations)])
        reordered_masks = np.array([row[perm] for row, perm in zip(pruned_masks, permutations)])
        reordered_active_masks = np.array([row[perm] for row, perm in zip(pruned_active_masks, permutations)])
        reordered_advantages = np.array([row[perm] for row, perm in zip(pruned_advantages, permutations)])    
        reordered_returns = np.array([row[perm] for row, perm in zip(pruned_returns, permutations)])
        reordered_active_agents = np.array([row[perm] for row, perm in zip(pruned_active_agents, permutations)])
        reordered_agent_groups = np.array([row[perm] for row, perm in zip(pruned_agent_groups, permutations)])
        reordered_agent_groups = np.array([row[perm] for row, perm in zip(np.transpose(reordered_agent_groups, (0,2,1)), permutations)])

        if self.available_actions is not None:
            reordered_available_actions = np.array([row[perm] for row, perm in zip(pruned_available_actions, permutations)])

        reordered_rnn_states = np.array([row[perm] for row, perm in zip(pruned_rnn_states, permutations)])
        reordered_rnn_states_critic = np.array([row[perm] for row, perm in zip(pruned_rnn_states_critic, permutations)])
        
        if mini_batch_size is None:
            assert pruned_batch_size >= num_mini_batch, (
                "PPO requires the number of processes ({}) "
                "* number of steps ({}) = {} "
                "to be greater than or equal to the number of PPO mini batches ({})."
                "".format(n_rollout_threads, episode_length,
                          n_rollout_threads * episode_length,
                          num_mini_batch))
            mini_batch_size = pruned_batch_size // num_mini_batch

        rand = torch.randperm(pruned_batch_size).numpy()
        sampler = [rand[i * mini_batch_size:(i + 1) * mini_batch_size] for i in range(num_mini_batch)] #an array of randomized indices

        for indices in sampler:
            # [L,T,N,Dim]-->[L*T,N,Dim]-->[index,N,Dim]-->[index*N, Dim]
            # first randomizes the order of the batch of threads and steps, then batches over agents too
            if self._mixed_obs:
                share_obs_batch = {}
                obs_batch = {}
                for key in obs.keys():
                    obs_batch[key] = reordered_obs[key][indices].reshape(-1, *reordered_obs[key].shape[2:])
                    share_obs_batch[key] = reordered_share_obs[key][indices].reshape(-1, *reordered_share_obs[key].shape[2:])
            else: #not used
                share_obs_batch = reordered_share_obs[indices].reshape(-1, *reordered_share_obs.shape[2:])
                obs_batch = reordered_obs[indices].reshape(-1, *reordered_obs.shape[2:])
            
            actions_batch = reordered_actions[indices].reshape(-1, *reordered_actions.shape[2:])
            old_action_log_probs_batch = reordered_action_log_probs[indices].reshape(-1, *reordered_action_log_probs.shape[2:])
            value_preds_batch = reordered_value_preds[indices].reshape(-1, *reordered_value_preds.shape[2:])
            masks_batch = reordered_masks[indices].reshape(-1, *reordered_masks.shape[2:])
            active_masks_batch = reordered_active_masks[indices].reshape(-1, *reordered_active_masks.shape[2:])
            if advantages is None:
                adv_targ = None
            else:
                adv_targ = reordered_advantages[indices].reshape(-1, *reordered_advantages.shape[2:])
            return_batch = reordered_returns[indices].reshape(-1, *reordered_returns.shape[2:])
            
            active_agents_batch = reordered_active_agents[indices].reshape(-1, *reordered_returns.shape[2:])
            agent_groups_batch = reordered_agent_groups[indices]
            # agent_groups_batch = reordered_agent_groups[indices].reshape(-1, *reordered_agent_groups.shape[2:])

            if self.available_actions is not None:
                available_actions_batch = reordered_available_actions[indices].reshape(-1, *reordered_available_actions.shape[2:])
            else:
                available_actions_batch = None
            
            rnn_states_batch = reordered_rnn_states[indices].reshape(-1, *reordered_rnn_states.shape[2:])
            rnn_states_critic_batch = reordered_rnn_states_critic[indices].reshape(-1, *reordered_rnn_states_critic.shape[2:])
            
            yield share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, \
                  value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, \
                  adv_targ, available_actions_batch, active_agents_batch, agent_groups_batch

    def feed_forward_generator_transformer(self, advantages, num_mini_batch=None, mini_batch_size=None):
        """
        Yield training data for MLP policies.
        :param advantages: (np.ndarray) advantage estimates.
        :param num_mini_batch: (int) number of minibatches to split the batch into.
        :param mini_batch_size: (int) number of samples in each minibatch.
        """
        # episode_length, n_rollout_threads, num_agents = self.rewards.shape[0:3]
        episode_length = self.episode_length
        n_rollout_threads = self.n_rollout_threads
        num_agents = self.num_agents

        batch_size = n_rollout_threads * episode_length

        if mini_batch_size is None:
            assert batch_size >= num_mini_batch, (
                "PPO requires the number of processes ({}) "
                "* number of steps ({}) = {} "
                "to be greater than or equal to the number of PPO mini batches ({})."
                "".format(n_rollout_threads, episode_length,
                          n_rollout_threads * episode_length,
                          num_mini_batch))
            mini_batch_size = batch_size // num_mini_batch

        rand = torch.randperm(batch_size).numpy()
        sampler = [rand[i * mini_batch_size:(i + 1) * mini_batch_size] for i in range(num_mini_batch)]
        rows, cols = _shuffle_agent_grid(batch_size, num_agents)

        # keep (num_agent, dim)
        if self._mixed_obs:
            share_obs = {}
            obs = {}
            for key in self.share_obs.keys():
                share_obs[key] = self.share_obs[key][:-1].reshape(-1, *self.share_obs[key].shape[2:])
                share_obs[key] = share_obs[key][rows, cols]
            for key in self.obs.keys():
                obs[key] = self.obs[key][:-1].reshape(-1, *self.obs[key].shape[2:]) # merges the first two dimensions (episode_length, n_rollout_threads)
                obs[key] = obs[key][rows, cols]
        else:
            share_obs = self.share_obs[:-1].reshape(-1, *self.share_obs.shape[2:])
            share_obs = share_obs[rows, cols]
            obs = self.obs[:-1].reshape(-1, *self.obs.shape[2:])
            obs = obs[rows, cols]
        
        rnn_states = self.rnn_states[:-1].reshape(-1, *self.rnn_states.shape[2:])
        rnn_states = rnn_states[rows, cols]
        rnn_states_critic = self.rnn_states_critic[:-1].reshape(-1, *self.rnn_states_critic.shape[2:])
        rnn_states_critic = rnn_states_critic[rows, cols]
        actions = self.actions.reshape(-1, *self.actions.shape[2:])
        actions = actions[rows, cols]
        if self.available_actions is not None:
            available_actions = self.available_actions[:-1].reshape(-1, *self.available_actions.shape[2:])
            available_actions = available_actions[rows, cols]
        value_preds = self.value_preds[:-1].reshape(-1, *self.value_preds.shape[2:])
        value_preds = value_preds[rows, cols]
        returns = self.returns[:-1].reshape(-1, *self.returns.shape[2:])
        returns = returns[rows, cols]
        masks = self.masks[:-1].reshape(-1, *self.masks.shape[2:])
        masks = masks[rows, cols]
        active_masks = self.active_masks[:-1].reshape(-1, *self.active_masks.shape[2:])
        active_masks = active_masks[rows, cols]
        action_log_probs = self.action_log_probs.reshape(-1, *self.action_log_probs.shape[2:])
        action_log_probs = action_log_probs[rows, cols]
        advantages = advantages.reshape(-1, *advantages.shape[2:])
        advantages = advantages[rows, cols]
        agent_groups = self.agent_groups[:-1].reshape(-1, *self.agent_groups.shape[2:])
        agent_groups = agent_groups[rows, cols]

        for indices in sampler:
            # [L,T,N,Dim]-->[L*T,N,Dim]-->[index,N,Dim]-->[index*N, Dim] (batches over steps, threads and agents)
            if self._mixed_obs:
                share_obs_batch = {}
                obs_batch = {}
                for key in share_obs.keys():
                    share_obs_batch[key] = share_obs[key][indices].reshape(-1, *share_obs[key].shape[2:])
                for key in obs.keys():
                    obs_batch[key] = obs[key][indices].reshape(-1, *obs[key].shape[2:])
            else:
                share_obs_batch = share_obs[indices].reshape(-1, *share_obs.shape[2:])
                obs_batch = obs[indices].reshape(-1, *obs.shape[2:])
            rnn_states_batch = rnn_states[indices].reshape(-1, *rnn_states.shape[2:])
            rnn_states_critic_batch = rnn_states_critic[indices].reshape(-1, *rnn_states_critic.shape[2:])
            actions_batch = actions[indices].reshape(-1, *actions.shape[2:])
            if self.available_actions is not None:
                available_actions_batch = available_actions[indices].reshape(-1, *available_actions.shape[2:])
            else:
                available_actions_batch = None
            value_preds_batch = value_preds[indices].reshape(-1, *value_preds.shape[2:])
            return_batch = returns[indices].reshape(-1, *returns.shape[2:])
            masks_batch = masks[indices].reshape(-1, *masks.shape[2:])
            active_masks_batch = active_masks[indices].reshape(-1, *active_masks.shape[2:])
            old_action_log_probs_batch = action_log_probs[indices].reshape(-1, *action_log_probs.shape[2:])
            if advantages is None:
                adv_targ = None
            else:
                adv_targ = advantages[indices].reshape(-1, *advantages.shape[2:])
            agent_groups_batch = agent_groups[indices]

            yield share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, \
                  value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, \
                  adv_targ, available_actions_batch, agent_groups_batch
    