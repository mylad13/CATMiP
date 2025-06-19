import json
import time
import wandb
import os
import copy
import numpy as np
from itertools import chain
import torch
import imageio
from icecream import ic
import matplotlib.pyplot as plt
import cv2
from collections import defaultdict, deque
from hetmarl.utils.util import update_linear_schedule, get_shape_from_act_space, get_shape_from_obs_space, AsynchControl, plot_macro_obs, inject_noise_to_occupancy
from hetmarl.runner.shared.base_runner import Runner
import torch.nn as nn


def _t2n(x): #tensor to numpy array
    return x.detach().cpu().numpy()


class GridWorldRunner(Runner):
    def __init__(self, config):
        super(GridWorldRunner, self).__init__(config)
        self.init_hyperparameters()
        self.init_map_variables()
        self.init_keys()


    def get_available_actions(self,envs):
        return envs.get_available_actions()

    def correct_ma_bounds(self, macro_action):
        return np.clip(macro_action, 0, self.map_size - 1)
        
    def run(self):
        
        if self.asynch:
            def generate_random_period(min_t,max_t):
                return np.random.randint(min_t, max_t)
            self.asynch_control = AsynchControl(num_envs=self.n_rollout_threads, num_agents=self.num_agents,
                                                limit=self.episode_length, random_fn=generate_random_period, min_wait=2, max_wait=5, rest_time = 10)

        start = time.time()
        episodes = int(self.num_env_steps) // self.max_steps // self.n_rollout_threads
        
        for episode in range(episodes):
            self.init_env_info()
            self.init_map_variables()
            period_rewards = np.zeros((self.n_rollout_threads, self.num_agents, 1))
            auc_area = np.zeros((self.n_rollout_threads, self.max_steps), dtype=np.float32)
            auc_single_area = np.zeros((self.n_rollout_threads, self.num_agents, self.max_steps), dtype=np.float32)
            self.done_envs = np.zeros((self.n_rollout_threads,)).astype(bool)
            is_last_step = np.full((self.n_rollout_threads, self.num_agents), False)
            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode, episodes)

            if self.asynch:
                self.asynch_control.reset()

            values, actions, action_log_probs, rnn_states, rnn_states_critic = self.warmup()

            async_global_step = 0

            for step in range(self.max_steps):
                local_step = step % self.local_step_num
                global_step = (step // self.local_step_num) % self.episode_length
                
                actions_env = self.envs.get_short_term_action(self.macro_action)

                
                # number_of_rubbles_known = self.number_of_rubbles_known.copy()


                # Obser reward and next obs
                dict_obs, rewards, dones, infos = self.envs.step(actions_env) #dones seems to be broken when env is done early, so we manually set dones.
                
                
                self.connected_agent_groups = []
                for e in range(self.n_rollout_threads):
                    for key in self.sum_env_info_keys:
                        if not self.done_envs[e]:
                            if key in infos[e].keys():
                                self.env_info['sum_{}'.format(key)][e] += np.array(infos[e][key])
                            elif key == 'agent_reward':
                                self.env_info['sum_{}'.format(key)][e] += np.array(rewards[e].squeeze())
                            else:
                                self.env_info['avg_{}'.format(key)][e] += np.array(rewards[e].sum()/self.num_agents)
                    for key in self.equal_env_info_keys:
                        if key == 'merge_explored_ratio':
                            auc_area[e, step] = np.array(infos[e][key])
                            if np.all(dones[e]):
                                self.env_info[key][e] = infos[e][key]
                        if key == 'agent_explored_ratio':
                            auc_single_area[e, :, step] = np.array(infos[e][key])
                            if np.all(dones[e]):
                                self.env_info[key][e] = infos[e][key]
                        elif key in infos[e].keys():
                            if key == 'explored_ratio_step':
                                for agent_id in range(self.num_agents):
                                    agent_k = "agent{}_{}".format(agent_id, key)
                                    if agent_k in infos[e].keys():
                                        self.env_info[key][e][agent_id] = infos[e][agent_k]
                            else:
                                self.env_info[key][e] = infos[e][key]
                    for key in self.target_info_keys:
                        if key == 'target_found': #this is buggy
                            self.env_info[key][e] = int(infos[e][key].any())
                        elif infos[e][key]:
                            self.env_info[key][e] = infos[e][key]
                    for key in self.rubble_info_keys:
                        if key in infos[e].keys():
                            self.env_info[key][e] = max(self.env_info[key][e],infos[e][key])
                    if self.env_info["target_rescued"][e]:
                        dones[e] = True
                        self.done_envs[e] = True
                    # self.number_of_rubbles_known[e] = infos[e]['agent_number_of_known_rubbles']

                    self.agent_groups[e] = infos[e]['agent_groups']                    
                    self.connected_agent_groups.append(infos[e]['connected_agent_groups'])
                    self.agent_pos[e] = infos[e]['agent_pos']
                    self.agent_alive[e] = infos[e]['agent_alive']
                    

                if self.asynch:
                    self.asynch_control.step()
                    for e in range(self.n_rollout_threads):
                        for a in range(self.num_agents):
                            if self.agent_alive[e][a] == 0:
                                self.asynch_control.standby[e, a] = 0
                                self.asynch_control.active[e, a] = 0
                                is_last_step[e,a] = True
                                continue
                            if not self.asynch_control.active[e, a] and not self.asynch_control.standby[e, a]:
                                # Checks if agent has reached its short term goal and stopped and puts it on standby if so
                                #TODO: If the agent has reached its ultimate objective, instead of putting it on standby, it should be completely deactivated

                                # Under the assumption of full communication in training, this is added to ensure the final MA of all agents finish at the same time
                                # (Useful for when the only rewards are team rewards)
                                if self.env_info["target_rescued"][e] and not is_last_step[e,a]:
                                    is_last_step[e,a] = True
                                    # self.asynch_control.standby[e, a] = 1
                                    self.asynch_control.activate(e, a)

                                elif actions_env[e, a] == 3:
                                    self.asynch_control.standby[e, a] = 1
                                    # self.asynch_control.activate(e, a)
                                    # rewards[e, a] -= 5
                                    # self.env_info['sum_agent_reward'][e,a] -= 5
                                    # self.env_info['avg_total_reward'][e] -= 5
                                
                                elif actions_env[e, a] == 4:
                                    self.asynch_control.standby[e, a] = 1
                                    # self.asynch_control.activate(e, a)
                                    # rewards[e, a] -= 5
                                    # self.env_info['sum_agent_reward'][e,a] -= 5
                                    # self.env_info['avg_total_reward'][e] -= 5
                                    
                                elif self.target_found[e, a] == 0:
                                    if infos[e]['target_found'][a] == 1:
                                        self.target_found[e, a] = 1
                                        self.asynch_control.standby[e, a] = 1
                                        # self.asynch_control.activate(e, a)

                                # # if self.agent_types_list[a] == 2 and number_of_rubbles_known[e,a] != self.number_of_rubbles_known[e,a] \
                                # elif number_of_rubbles_known[e,a] != self.number_of_rubbles_known[e,a]:
                                #     self.asynch_control.standby[e, a] = 1
                                #     # self.asynch_control.activate(e, a)
                    
                    
                    if np.any(self.asynch_control.standby) and np.any(self.asynch_control.active): #Activates on-standby agents based on communication model
                        for thread in self.asynch_control.active_agents_threads():
                            if len(thread) > 1:
                                if self.use_partial_comm:
                                    connected_agents = infos[thread[0]]['connected_agent_groups']
                                    for agent_id in thread[1:]:
                                        for group in connected_agents:
                                            if agent_id in group:
                                                for i in group:
                                                    if self.asynch_control.standby[thread[0], i] and self.asynch_control.wait[thread[0], i] <= generate_random_period(2, 5):
                                                        self.asynch_control.activate(thread[0], i)
                                elif self.use_full_comm:
                                    for i in range(self.num_agents):
                                        if self.asynch_control.standby[thread[0], i] and self.asynch_control.wait[thread[0], i] <= generate_random_period(2, 5):
                                            self.asynch_control.activate(thread[0], i)

                period_rewards += rewards
            
                if (not self.asynch and local_step == self.local_step_num - 1) or (self.asynch and np.any(self.asynch_control.active)):
                
                    if self.use_action_masking:
                        available_actions = self.get_available_actions(self.envs)
                        self.available_actions = available_actions
                    else:
                        available_actions = None
                        self.available_actions = None

                    data = dict_obs, period_rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic, self.agent_groups, available_actions
                    
                    # insert data into buffer
                    if not self.asynch:
                        self.insert(data, step)
                        period_rewards = np.zeros((self.n_rollout_threads, self.num_agents, 1))
                    else:
                        self.insert(data, step, active_agents=self.asynch_control.active_agents())
                        for e, a, s in self.asynch_control.active_agents():
                            period_rewards[e, a, 0] = 0.

                    if not self.asynch:
                        values, actions, action_log_probs, rnn_states, rnn_states_critic = self.compute_global_goal(step=global_step + 1)
                    else:
                        async_values, async_actions, async_action_log_probs, async_rnn_states, async_rnn_states_critic = self.compute_global_goal(step=async_global_step + 1)
                        active_mask = (self.asynch_control.active == 1)
                        
                        
                        values[active_mask] = async_values[active_mask]
                        actions[active_mask] = async_actions[active_mask]
                        action_log_probs[active_mask] = async_action_log_probs[active_mask]
                        rnn_states[active_mask] = async_rnn_states[active_mask]
                        rnn_states_critic[active_mask] = async_rnn_states_critic[active_mask]
                        async_global_step += 1

                if np.all(dones): # If all envs are done, finish the episode
                    break
            
            # compute returns and update the network
            if self.asynch:
                self.buffer.update_mask(self.asynch_control.cnt)
            else:
                self.buffer.active_masks = self.buffer.masks # to ensure correct computation of returns
            self.compute(values)
            train_infos = self.train()

            # post process
            total_num_steps = (episode + 1) * self.max_steps * self.n_rollout_threads

            # save model
            if (episode % self.save_interval == 0 or episode == episodes - 1):
                if episode % 100*self.save_interval == 0:
                    self.save(episode, save_separately=True)
                else:
                    self.save(episode, save_separately=False)

            # log information
            self.convert_info()
            
            
            if episode % self.log_interval == 0:
                end = time.time()
                print("\n Scenario {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n"
                      .format(self.all_args.scenario_name,
                              self.algorithm_name,
                              self.experiment_name,
                              episode,
                              episodes,
                              total_num_steps,
                              self.num_env_steps,
                              int(total_num_steps / (end - start))))

                self.log_env(self.env_infos, total_num_steps)
                # self.log_agent(self.env_infos, total_num_steps)
                self.log_train(train_infos, total_num_steps)
                
            # eval
            if episode % self.eval_interval == 0 and self.use_eval: # not recommended, might not be working
               self.eval(total_num_steps)


    def _convert(self, dict_obs, infos, step, active_agents=None):
        obs = {}
        obs['agent_class_identifier'] = np.zeros((len(dict_obs), self.num_agents, self.n_agent_types), dtype=int)
        # obs['agent_inventory'] = np.zeros((len(dict_obs), self.num_agents, 2), dtype=int)
        obs['global_agent_map'] = np.zeros((len(dict_obs), self.num_agents, 7, self.map_size, self.map_size), dtype=np.float32)
        obs['local_agent_map'] = np.zeros((len(dict_obs), self.num_agents, 6, 2*self.action_size + 1, 2*self.action_size + 1), dtype=np.float32)
        if self.algorithm_name == 'mancp':
            obs['timespan'] = np.zeros((len(dict_obs), self.num_agents, 1), dtype=np.float32)
            # active agents are [e,a,cnt]
            if active_agents is not None:
                for e, a, cnt in active_agents:
                    obs['timespan'][e, a] = step - self.last_active_step[e, a]
                    self.last_active_step[e, a] = step

        explored_map = np.zeros((len(dict_obs), self.num_agents, self.full_w, self.full_h), dtype=np.float32)
        occupancy_map = np.zeros((len(dict_obs), self.num_agents, self.full_w, self.full_h), dtype=np.float32)
        noisy_occupancy_map = np.zeros((len(dict_obs), self.num_agents, self.full_w, self.full_h), dtype=np.float32)

        current_agent_pos = np.zeros((len(dict_obs), self.num_agents, 2), dtype=np.int32)
        agent_pos_map = np.zeros((len(dict_obs), self.num_agents, self.full_w, self.full_h), dtype=np.float32)
        global_agent_pos_map = np.zeros((len(dict_obs), self.num_agents, self.full_w, self.full_h), dtype=np.float32)
        global_type0_agent_pos_map = np.zeros((len(dict_obs), self.num_agents, self.full_w, self.full_h), dtype=np.float32) #other agents of type 0
        global_type1_agent_pos_map = np.zeros((len(dict_obs), self.num_agents, self.full_w, self.full_h), dtype=np.float32) #other agents of type 1
        # global_type2_agent_pos_map = np.zeros((len(dict_obs), self.num_agents, self.full_w, self.full_h), dtype=np.float32) #other agents of type 2
        type0_agent_pos_map = np.zeros((len(dict_obs), self.num_agents, self.full_w, self.full_h), dtype=np.float32) #other agents of type 0
        type1_agent_pos_map = np.zeros((len(dict_obs), self.num_agents, self.full_w, self.full_h), dtype=np.float32) #other agents of type 1
        type2_agent_pos_map = np.zeros((len(dict_obs), self.num_agents, self.full_w, self.full_h), dtype=np.float32) #other agents of type 2
        # other_agents_pos_map = np.zeros((len(dict_obs), self.num_agents, self.full_w, self.full_h), dtype=np.float32)
        
        self.agent_goal_history = np.zeros((self.n_rollout_threads, self.num_agents, self.full_w, self.full_h), dtype=np.float32)
        # target_all_map = np.zeros((self.n_rollout_threads, self.full_w, self.full_h), dtype=np.float32)
        # rubble_all_map = np.zeros((self.n_rollout_threads, self.full_w, self.full_h), dtype=np.float32)
        # target_each_map = np.zeros((self.n_rollout_threads, self.num_agents, self.full_w, self.full_h), dtype=np.float32)
        # rubble_each_map = np.zeros((self.n_rollout_threads, self.num_agents, self.full_w, self.full_h), dtype=np.float32)

        # # Map of agent direction indices to vectors
        # DIR_TO_VEC = [
        #     # Pointing right (positive X)
        #     np.array((1, 0)),
        #     # Down (positive Y)
        #     np.array((0, 1)),
        #     # Pointing left (negative X)
        #     np.array((-1, 0)),
        #     # Up (negative Y)
        #     np.array((0, -1)),
        # ]
        

        if self.use_full_comm:
            for e in range(len(dict_obs)):
                for agent_id in range(self.num_agents):
                    
                    explored_map[e,agent_id] = infos[e]['explored_all_map']
                    occupancy_map[e, agent_id] = infos[e]['occupied_all_map']
                    explored_mask = explored_map[e, agent_id] == 1
                    noisy_occupancy_map[e, agent_id] = inject_noise_to_occupancy(occupancy_map[e, agent_id], explored_mask, p_fp=0.02, p_fn=0.05)


                    # agent_dir_vec = DIR_TO_VEC[infos[e]['agent_direction'][agent_id]]
                    # current_agent_pos is based on the larger full map
                    current_agent_pos[e, agent_id] = infos[e]['current_agent_pos'][agent_id]
                    agent_pos_map[e, agent_id, current_agent_pos[e,agent_id][0],
                                   current_agent_pos[e,agent_id][1]] = 1
                    # global_agent_pos_map[e, agent_id, current_agent_pos[e,agent_id][0]-1:current_agent_pos[e,agent_id][0]+2,
                    #                current_agent_pos[e,agent_id][1]-1:current_agent_pos[e,agent_id][1]+2] = 1
                    global_agent_pos_map[e, agent_id, current_agent_pos[e,agent_id][0],
                                   current_agent_pos[e,agent_id][1]] = 1
                    # for target_cell in infos[e]['all_target_set']:
                    #     target_all_map[e, target_cell[1]+self.agent_view_size-1: target_cell[1]+self.agent_view_size+2,
                    #                 target_cell[0]+self.agent_view_size-1:target_cell[0]+self.agent_view_size+2] = 1
                    # for rubble_cell in infos[e]['all_rubble_set']:
                    #     rubble_all_map[e, rubble_cell[1]+self.agent_view_size-1: rubble_cell[1]+self.agent_view_size+2,
                    #                 rubble_cell[0]+self.agent_view_size-1:rubble_cell[0]+self.agent_view_size+2] = 1

                    # agent_pos_map[e, agent_id, infos[e]['current_agent_pos'][agent_id][0] + agent_dir_vec[1],
                    #                             infos[e]['current_agent_pos'][agent_id][1] + agent_dir_vec[0]] = 0.75
                    # highlighted_action_area[e,agent_id, self.agent_pos[e, agent_id][1]:self.agent_pos[e, agent_id][1] + 2*self.action_size + 1,
                    #                         self.agent_pos[e, agent_id][0]:self.agent_pos[e, agent_id][0] + 2*self.action_size + 1] = 1 
                    self.agent_goal_history[e,agent_id] *= self.goal_history_decay
                    self.agent_goal_history[e,agent_id, self.macro_action[e, agent_id][1]+self.agent_view_size,
                                            self.macro_action[e, agent_id][0]+self.agent_view_size] = 1
                    self.agent_goal_history[e,agent_id,0,0] = 0                   

                for agent_id in range(self.num_agents):
                    for j in range(self.num_agents):
                        if j != agent_id:
                            # other_agents_pos_map[e,agent_id] = np.maximum(other_agents_pos_map[e,agent_id],agent_pos_map[e,j])
                            if self.agent_types_list[e,j] == 0:
                                type0_agent_pos_map[e,agent_id] = np.maximum(type0_agent_pos_map[e,agent_id],agent_pos_map[e,j])
                                global_type0_agent_pos_map[e,agent_id] = np.maximum(global_type0_agent_pos_map[e,agent_id],global_agent_pos_map[e,j])
                            elif self.agent_types_list[e,j] == 1:
                                type1_agent_pos_map[e,agent_id] = np.maximum(type1_agent_pos_map[e,agent_id],agent_pos_map[e,j])
                                global_type1_agent_pos_map[e,agent_id] = np.maximum(global_type1_agent_pos_map[e,agent_id],global_agent_pos_map[e,j])
                            # elif self.agent_types_list[e,j] == 2:
                            #     type2_agent_pos_map[e,agent_id] = np.maximum(type2_agent_pos_map[e,agent_id],agent_pos_map[e,j])
                            #     global_type2_agent_pos_map[e,agent_id] = np.maximum(global_type2_agent_pos_map[e,agent_id],global_agent_pos_map[e,j])

            for e in range(len(dict_obs)):
                for agent_id in range(self.num_agents):
                    if self.agent_alive[e][agent_id] == 0:
                        continue

                    obs['global_agent_map'][e, agent_id, 0] = infos[e]['explored_all_map'][self.agent_view_size:self.full_w -
                                                                    self.agent_view_size, self.agent_view_size:self.full_w-self.agent_view_size] 
                    obs['global_agent_map'][e, agent_id, 1] = noisy_occupancy_map[e, agent_id][self.agent_view_size:self.full_w -
                                                                    self.agent_view_size, self.agent_view_size:self.full_w-self.agent_view_size]
                    obs['global_agent_map'][e, agent_id, 2] = infos[e]['global_target_all_map'][self.agent_view_size:self.full_w -
                                                                    self.agent_view_size, self.agent_view_size:self.full_w-self.agent_view_size]
                    # obs['global_agent_map'][e, agent_id, 3] = infos[e]['global_rubble_all_map'][self.agent_view_size:self.full_w -
                    #                                                     self.agent_view_size, self.agent_view_size:self.full_w-self.agent_view_size]
                    # obs['global_agent_map'][e, agent_id, 3] = infos[e]["each_agent_trajectory_map"][agent_id][self.agent_view_size:self.full_w - self.agent_view_size,
                    #                                                 self.agent_view_size:self.full_h - self.agent_view_size]
                    obs['global_agent_map'][e, agent_id, 3] = global_agent_pos_map[e, agent_id][self.agent_view_size:self.full_w -
                                                                    self.agent_view_size, self.agent_view_size:self.full_w-self.agent_view_size]
                    obs['global_agent_map'][e, agent_id, 4] = global_type0_agent_pos_map[e,agent_id][self.agent_view_size:self.full_w -
                                                                    self.agent_view_size, self.agent_view_size:self.full_w-self.agent_view_size]
                    obs['global_agent_map'][e, agent_id, 5] = global_type1_agent_pos_map[e,agent_id][self.agent_view_size:self.full_w -
                                                                    self.agent_view_size, self.agent_view_size:self.full_w-self.agent_view_size]
                    # obs['global_agent_map'][e, agent_id, 7] = global_type2_agent_pos_map[e,agent_id][self.agent_view_size:self.full_w -
                    #                                                                self.agent_view_size, self.agent_view_size:self.full_w-self.agent_view_size]
                    obs['global_agent_map'][e, agent_id, 6] = self.agent_goal_history[e,agent_id][self.agent_view_size:self.full_w -
                                                                    self.agent_view_size, self.agent_view_size:self.full_w-self.agent_view_size]
                    # obs['global_agent_map'][e, agent_id, 6] = other_agents_pos_map[e,agent_id][self.agent_view_size:self.full_w -
                    #                                                 self.agent_view_size, self.agent_view_size:self.full_w-self.agent_view_size]
                                       
                    obs['local_agent_map'][e,agent_id,0] = infos[e]['explored_all_map'][current_agent_pos[e, agent_id][0]-self.action_size:current_agent_pos[e, agent_id][0]
                                                                +self.action_size+1, current_agent_pos[e, agent_id][1]-self.action_size:current_agent_pos[e, agent_id][1]+self.action_size+1]
                    obs['local_agent_map'][e,agent_id,1] = noisy_occupancy_map[e, agent_id][current_agent_pos[e, agent_id][0]-self.action_size:current_agent_pos[e, agent_id][0]
                                                                +self.action_size+1, current_agent_pos[e, agent_id][1]-self.action_size:current_agent_pos[e, agent_id][1]+self.action_size+1]
                    obs['local_agent_map'][e,agent_id,2] = infos[e]['target_all_map'][current_agent_pos[e, agent_id][0]-self.action_size:current_agent_pos[e, agent_id][0]
                                                                +self.action_size+1, current_agent_pos[e, agent_id][1]-self.action_size:current_agent_pos[e, agent_id][1]+self.action_size+1]
                    # obs['local_agent_map'][e,agent_id,3] = infos[e]['rubble_all_map'][current_agent_pos[e, agent_id][0]-self.action_size:current_agent_pos[e, agent_id][0]
                    #                                             +self.action_size+1, current_agent_pos[e, agent_id][1]-self.action_size:current_agent_pos[e, agent_id][1]+self.action_size+1]
                    obs['local_agent_map'][e,agent_id,3] = type0_agent_pos_map[e,agent_id][current_agent_pos[e, agent_id][0]-self.action_size:current_agent_pos[e, agent_id][0]
                                                                +self.action_size+1, current_agent_pos[e, agent_id][1]-self.action_size:current_agent_pos[e, agent_id][1]+self.action_size+1]
                    obs['local_agent_map'][e,agent_id,4] = type1_agent_pos_map[e,agent_id][current_agent_pos[e, agent_id][0]-self.action_size:current_agent_pos[e, agent_id][0]
                                                                +self.action_size+1, current_agent_pos[e, agent_id][1]-self.action_size:current_agent_pos[e, agent_id][1]+self.action_size+1]
                    # obs['local_agent_map'][e,agent_id,6] = type2_agent_pos_map[e,agent_id][current_agent_pos[e, agent_id][0]-self.action_size:current_agent_pos[e, agent_id][0]
                    #                                             +self.action_size+1, current_agent_pos[e, agent_id][1]-self.action_size:current_agent_pos[e, agent_id][1]+self.action_size+1]
                    obs['local_agent_map'][e,agent_id,5] = self.agent_goal_history[e,agent_id][current_agent_pos[e, agent_id][0]-self.action_size:current_agent_pos[e, agent_id][0]
                                                                +self.action_size+1, current_agent_pos[e, agent_id][1]-self.action_size:current_agent_pos[e, agent_id][1]+self.action_size+1]
                    
                    # obs['local_agent_map'][e,agent_id,3] = infos[e]['each_agent_trajectory_map'][agent_id][current_agent_pos[e, agent_id][0]-self.action_size:current_agent_pos[e, agent_id][0]
                    #                                             +self.action_size+1, current_agent_pos[e, agent_id][1]-self.action_size:current_agent_pos[e, agent_id][1]+self.action_size+1]
                    
                
                obs['agent_class_identifier'][e] = self.agent_class_identifier[e]
                # obs['agent_inventory'][e] = infos[e]['agent_inventory']
            # obs['global_agent_map'] = np.zeros((len(dict_obs), self.num_agents, 8, self.map_size, self.map_size), dtype=np.float32)
            # obs['local_agent_map'] = np.zeros((len(dict_obs), self.num_agents, 4, 2*self.action_size + 1, 2*self.action_size + 1), dtype=np.float32)
        elif self.use_partial_comm:
            for e in range(len(dict_obs)):
                for agent_id in range(self.num_agents):                   
                    # agent_dir_vec = DIR_TO_VEC[infos[e]['agent_direction'][agent_id]]

                    current_agent_pos[e, agent_id] = infos[e]['current_agent_pos'][agent_id]
                    agent_pos_map[e, agent_id, current_agent_pos[e,agent_id][0], current_agent_pos[e,agent_id][1]] = 1
                    global_agent_pos_map[e, agent_id, current_agent_pos[e,agent_id][0]-1:current_agent_pos[e,agent_id][0]+2,
                                   current_agent_pos[e,agent_id][1]-1:current_agent_pos[e,agent_id][1]+2] = 1
                    # agent_pos_map[e, agent_id, infos[e]['current_agent_pos'][agent_id][0] + agent_dir_vec[1],
                    #                             infos[e]['current_agent_pos'][agent_id][1] + agent_dir_vec[0]] = 0.75
                    # for trace_cell in infos[e]['agent_trace_sets'][agent_id]:
                    #     target_each_map[e, agent_id, trace_cell[1]+self.agent_view_size-1: trace_cell[1]+self.agent_view_size+2,
                    #                 trace_cell[0]+self.agent_view_size-1:trace_cell[0]+self.agent_view_size+2] = 0.25
                    # for target_cell in infos[e]['agent_target_sets'][agent_id]:
                    #     target_each_map[e, agent_id, target_cell[1]+self.agent_view_size-1: target_cell[1]+self.agent_view_size+2,
                    #                 target_cell[0]+self.agent_view_size-1:target_cell[0]+self.agent_view_size+2] = 1
                    # for rubble_cell in infos[e]['agent_rubble_sets'][agent_id]:
                    #     rubble_each_map[e, agent_id, rubble_cell[1]+self.agent_view_size-1: rubble_cell[1]+self.agent_view_size+2,
                    #                 rubble_cell[0]+self.agent_view_size-1:rubble_cell[0]+self.agent_view_size+2] = 1



                    self.agent_goal_history[e,agent_id] *= self.goal_history_decay
                    self.agent_goal_history[e,agent_id, self.macro_action[e, agent_id][1] + self.agent_view_size,
                                            self.macro_action[e, agent_id][0] + self.agent_view_size] = 1
                    self.agent_goal_history[e,agent_id,0,0] = 0
                    # for j in range(agent_id, self.num_agents):
                    #     if self.macro_action[e, agent_id][0] == self.macro_action[e, j][0] and self.macro_action[e, agent_id][1] == self.macro_action[e, j][1]:
                    #         goal_cell_value[e,agent_id] += 1/self.num_agents
                    
                connected_agent_groups = infos[e]['connected_agent_groups']
                counter = 0
                for group in connected_agent_groups:
                    for agent_id in group:
                        for j in group:
                            if j != agent_id:
                                # other_agents_pos_map[e,agent_id] = np.maximum(other_agents_pos_map[e,agent_id],agent_pos_map[e,j])
                                if self.agent_types_list[e,j] == 0:
                                    type0_agent_pos_map[e,agent_id] = np.maximum(type0_agent_pos_map[e,agent_id],agent_pos_map[e,j])
                                    global_type0_agent_pos_map[e,agent_id] = np.maximum(global_type0_agent_pos_map[e,agent_id],global_agent_pos_map[e,j])
                                elif self.agent_types_list[e,j] == 1:
                                    type1_agent_pos_map[e,agent_id] = np.maximum(type1_agent_pos_map[e,agent_id],agent_pos_map[e,j])
                                    global_type1_agent_pos_map[e,agent_id] = np.maximum(global_type1_agent_pos_map[e,agent_id],global_agent_pos_map[e,j])
                                # elif self.agent_types_list[e,j] == 2:
                                #     type2_agent_pos_map[e,agent_id] = np.maximum(type2_agent_pos_map[e,agent_id],agent_pos_map[e,j])
                                #     global_type2_agent_pos_map[e,agent_id] = np.maximum(global_type2_agent_pos_map[e,agent_id],global_agent_pos_map[e,j])
                                    
            for e in range(len(dict_obs)):
                for agent_id in range(self.num_agents):
                    if self.agent_alive[e][agent_id] == 0:
                        continue                    
                    obs['global_agent_map'][e, agent_id, 0] = infos[e]['explored_each_map'][agent_id][self.agent_view_size:self.full_w -
                                                                    self.agent_view_size, self.agent_view_size:self.full_w-self.agent_view_size] 
                    obs['global_agent_map'][e, agent_id, 1] = infos[e]['occupied_each_map'][agent_id][self.agent_view_size:self.full_w -
                                                                        self.agent_view_size, self.agent_view_size:self.full_w-self.agent_view_size]
                    obs['global_agent_map'][e, agent_id, 2] = infos[e]['global_target_each_map'][agent_id][self.agent_view_size:self.full_w -
                                                                    self.agent_view_size, self.agent_view_size:self.full_w-self.agent_view_size]
                    # obs['global_agent_map'][e, agent_id, 3] = infos[e]['global_rubble_each_map'][agent_id][self.agent_view_size:self.full_w -
                    #                                                     self.agent_view_size, self.agent_view_size:self.full_w-self.agent_view_size]
                    # obs['global_agent_map'][e, agent_id, 3] = infos[e]["each_agent_trajectory_map"][agent_id][self.agent_view_size:self.full_w - self.agent_view_size,
                    #                                                                         self.agent_view_size:self.full_h - self.agent_view_size]                    
                    obs['global_agent_map'][e, agent_id, 3] = global_agent_pos_map[e, agent_id][self.agent_view_size:self.full_w -
                                                                                   self.agent_view_size, self.agent_view_size:self.full_w-self.agent_view_size]
                    obs['global_agent_map'][e, agent_id, 4] = global_type0_agent_pos_map[e,agent_id][self.agent_view_size:self.full_w -
                                                                                   self.agent_view_size, self.agent_view_size:self.full_w-self.agent_view_size]
                    obs['global_agent_map'][e, agent_id, 5] = global_type1_agent_pos_map[e,agent_id][self.agent_view_size:self.full_w -
                                                                                   self.agent_view_size, self.agent_view_size:self.full_w-self.agent_view_size]
                    # obs['global_agent_map'][e, agent_id, 7] = global_type2_agent_pos_map[e,agent_id][self.agent_view_size:self.full_w -
                    #                                                                self.agent_view_size, self.agent_view_size:self.full_w-self.agent_view_size]
                    obs['global_agent_map'][e, agent_id, 6] = self.agent_goal_history[e,agent_id][self.agent_view_size:self.full_w -
                                                                    self.agent_view_size, self.agent_view_size:self.full_w-self.agent_view_size]
                    # obs['global_agent_map'][e, agent_id, 6] = other_agents_pos_map[e,agent_id][self.agent_view_size:self.full_w -
                    #                                                 self.agent_view_size, self.agent_view_size:self.full_w-self.agent_view_size]
                    # obs['global_agent_map'][e, agent_id, 5] = infos[e]["each_agent_trajectory_map"][agent_id][self.agent_view_size:self.full_w - self.agent_view_size,
                    #                                                                                             self.agent_view_size:self.full_h - self.agent_view_size]

                    obs['local_agent_map'][e,agent_id,0] = infos[e]['explored_each_map'][agent_id][current_agent_pos[e, agent_id][0]-self.action_size:current_agent_pos[e, agent_id][0]
                                                                +self.action_size+1, current_agent_pos[e, agent_id][1]-self.action_size:current_agent_pos[e, agent_id][1]+self.action_size+1]
                    obs['local_agent_map'][e,agent_id,1] = infos[e]['occupied_each_map'][agent_id][current_agent_pos[e, agent_id][0]-self.action_size:current_agent_pos[e, agent_id][0]
                                                                +self.action_size+1, current_agent_pos[e, agent_id][1]-self.action_size:current_agent_pos[e, agent_id][1]+self.action_size+1]
                    obs['local_agent_map'][e,agent_id,2] = infos[e]['target_each_map'][agent_id][current_agent_pos[e, agent_id][0]-self.action_size:current_agent_pos[e, agent_id][0]
                                                                +self.action_size+1, current_agent_pos[e, agent_id][1]-self.action_size:current_agent_pos[e, agent_id][1]+self.action_size+1]
                    # obs['local_agent_map'][e,agent_id,3] = infos[e]['rubble_each_map'][agent_id][current_agent_pos[e, agent_id][0]-self.action_size:current_agent_pos[e, agent_id][0]
                    #                                             +self.action_size+1, current_agent_pos[e, agent_id][1]-self.action_size:current_agent_pos[e, agent_id][1]+self.action_size+1]
                    obs['local_agent_map'][e,agent_id,3] = type0_agent_pos_map[e,agent_id][current_agent_pos[e, agent_id][0]-self.action_size:current_agent_pos[e, agent_id][0]
                                                                +self.action_size+1, current_agent_pos[e, agent_id][1]-self.action_size:current_agent_pos[e, agent_id][1]+self.action_size+1]
                    obs['local_agent_map'][e,agent_id,4] = type1_agent_pos_map[e,agent_id][current_agent_pos[e, agent_id][0]-self.action_size:current_agent_pos[e, agent_id][0]
                                                                +self.action_size+1, current_agent_pos[e, agent_id][1]-self.action_size:current_agent_pos[e, agent_id][1]+self.action_size+1]
                    # obs['local_agent_map'][e,agent_id,6] = type2_agent_pos_map[e,agent_id][current_agent_pos[e, agent_id][0]-self.action_size:current_agent_pos[e, agent_id][0]
                    #                                             +self.action_size+1, current_agent_pos[e, agent_id][1]-self.action_size:current_agent_pos[e, agent_id][1]+self.action_size+1]
                    obs['local_agent_map'][e,agent_id,5] = self.agent_goal_history[e,agent_id][current_agent_pos[e, agent_id][0]-self.action_size:current_agent_pos[e, agent_id][0]
                                                                +self.action_size+1, current_agent_pos[e, agent_id][1]-self.action_size:current_agent_pos[e, agent_id][1]+self.action_size+1]
                    # obs['local_agent_map'][e,agent_id,3] = infos[e]['each_agent_trajectory_map'][agent_id][current_agent_pos[e, agent_id][0]-self.action_size:current_agent_pos[e, agent_id][0]
                    #                                             +self.action_size+1, current_agent_pos[e, agent_id][1]-self.action_size:current_agent_pos[e, agent_id][1]+self.action_size+1]

                obs['agent_class_identifier'][e] = self.agent_class_identifier[e]
                # obs['agent_inventory'][e] = infos[e]['agent_inventory']
        else:
            pass
        # plt.imshow(obs['global_agent_map'][0,0,1])
        # plt.imshow(obs['local_agent_map'][0,0,2])
        # plt.show()
        # plot_macro_obs(obs, 0)

        return obs

    def warmup(self):
        # reset env
        dict_obs, infos = self.envs.reset()
        for e in range(self.n_rollout_threads):
            self.agent_pos[e] = infos[e]['agent_pos']
            self.agent_types_list[e] = np.array(infos[e]['agent_types_list'])
        
        # one-hot agent_class_identifier
        self.agent_class_identifier = np.zeros((self.n_rollout_threads, self.num_agents, self.n_agent_types), dtype=np.int32)
        for e in range(self.n_rollout_threads):
            for agent_id in range(self.num_agents):
                for agent_type in range(self.n_agent_types):
                    if self.agent_types_list[e, agent_id] == agent_type:
                        self.agent_class_identifier[e, agent_id, agent_type] = 1
        
        if self.use_action_masking:
            available_actions = self.get_available_actions(self.envs)
            self.available_actions = available_actions
        else:
            available_actions = None
            self.available_actions = None

        obs = self._convert(dict_obs, infos, 0)
        self.obs = obs
        # if not self.use_centralized_V:
        # share_obs = obs.copy()
        
        for key in obs.keys():
            self.buffer.obs[key][0] = obs[key].copy()
            self.buffer.all_obs[key][0] = obs[key].copy()

        # for key in share_obs.keys():
        #     self.buffer.share_obs[key][0] = share_obs[key].copy()

        if available_actions is not None:
            self.buffer.available_actions[0] = available_actions.copy()
        
        
        # used for training with partial comm
        self.connected_agent_groups = []
        for e in range(len(dict_obs)):
            self.connected_agent_groups.append(infos[e]['connected_agent_groups'])
            self.agent_groups[e] = infos[e]['agent_groups']
        self.buffer.agent_groups[0] = self.agent_groups.copy()


        
        values, actions, action_log_probs, rnn_states, rnn_states_critic = self.compute_global_goal(0)

        return values, actions, action_log_probs, rnn_states, rnn_states_critic


    def init_hyperparameters(self):
        # Calculating full and local map sizes
        self.map_size = self.all_args.grid_size
        self.max_steps = self.all_args.max_steps
        self.local_step_num = self.all_args.local_step_num
        self.agent_view_size = self.all_args.agent_view_size
        self.full_w, self.full_h = self.map_size + 2*self.agent_view_size, self.map_size + 2*self.agent_view_size
        
        self.use_action_masking = self.all_args.use_action_masking        

        self.asynch = self.all_args.asynch

        # function_parameters
        self.use_full_comm = self.all_args.use_full_comm
        self.use_partial_comm = self.all_args.use_partial_comm
        self.use_intrinsic_reward = self.all_args.use_intrinsic_reward
        self.use_orientation = self.all_args.use_orientation
        self.goal_history_decay = self.all_args.goal_history_decay
        self.path_prediction_decay = self.all_args.path_prediction_decay

        self.best_gobal_reward = -np.inf

        self.recurrent_hidden_size = self.all_args.recurrent_hidden_size
    
    def init_keys(self): #TODO: clean this up
        # info keys
        self.equal_env_info_keys = ['merge_explored_ratio']
        self.sum_env_info_keys = ['agent_explored_reward', 'total_reward', 'agent_reward']

        #log keys
        self.agents_env_info_keys = ['sum_agent_explored_reward','sum_agent_reward']
        self.env_info_keys = ['merge_explored_ratio','avg_total_reward']
            
        if self.use_eval:
            self.eval_env_info_keys = ['eval_merge_explored_ratio']
        
        self.target_info_keys = ['target_rescued','target_rescued_step','target_found','target_found_step','is_target_found']
        self.rubble_info_keys = ['number_of_rubbles_removed']

        # convert keys
        self.env_infos_keys = self.agents_env_info_keys + self.env_info_keys + self.target_info_keys + self.rubble_info_keys
        self.env_infos = {}
        for key in self.env_infos_keys:
            self.env_infos[key] = deque(maxlen=1)
    
    def init_env_info(self):
        self.env_info = {}

        for key in self.agents_env_info_keys:
            if "step" in key:
                self.env_info[key] = np.ones((self.n_rollout_threads, self.num_agents), dtype=np.float32) * self.max_steps
            else:
                self.env_info[key] = np.zeros((self.n_rollout_threads, self.num_agents), dtype=np.float32)
        
        for key in self.env_info_keys:
            if "step" in key:
                self.env_info[key] = np.ones((self.n_rollout_threads,), dtype=np.float32) * self.max_steps
            else:
                self.env_info[key] = np.zeros((self.n_rollout_threads,), dtype=np.float32)
        
        for key in self.target_info_keys:
            self.env_info[key] = np.zeros((self.n_rollout_threads,), dtype=np.float32)

        for key in self.rubble_info_keys:
            self.env_info[key] = np.zeros((self.n_rollout_threads,), dtype=np.float32)

    def init_eval_env_info(self):
        self.eval_env_info = {}
        for key in self.eval_env_info_keys:
            if "step" in key:
                self.eval_env_info[key] = np.ones((self.n_eval_rollout_threads,), dtype=np.float32) * self.max_steps
            else:
                self.eval_env_info[key] = np.zeros((self.n_eval_rollout_threads,), dtype=np.float32)
        
        for key in self.target_info_keys:
            self.eval_env_info[key] = np.zeros((self.n_eval_rollout_threads,), dtype=np.float32)
        
        for key in self.rubble_info_keys:
            self.eval_env_info[key] = np.zeros((self.n_rollout_threads,), dtype=np.float32)


    def init_map_variables(self):
        # action space
        # self.act_dim = self.envs.action_space[0].high - self.envs.action_space[0].low + 1 #for multidiscrete action space
        self.act_dim = self.envs.action_space[0].n #for discrete action space
        self.action_shape = get_shape_from_act_space(self.envs.action_space[0])
        if self.use_action_masking:
            # self.available_actions = np.ones((self.n_rollout_threads, self.num_agents, *self.act_dim), dtype=np.int32)
            self.available_actions = np.ones((self.n_rollout_threads, self.num_agents, self.act_dim), dtype=np.int32)
        else:
            self.available_actions = None
        
        # Initializing agent pos, groups and actions info
        self.macro_action = np.zeros((self.n_rollout_threads, self.num_agents, 2), dtype=np.int32)
        self.agent_pos = np.zeros((self.n_rollout_threads, self.num_agents, 2), dtype=np.int32) # based on the actual map
        self.agent_groups = np.ones((self.n_rollout_threads, self.num_agents, self.num_agents), dtype=np.int32)
        self.agent_alive = np.ones((self.n_rollout_threads, self.num_agents), dtype=np.int32)

        self.target_found = np.zeros((self.n_rollout_threads,self.num_agents), dtype=np.int32)
        # self.number_of_rubbles_known = np.zeros((self.n_rollout_threads, self.num_agents), dtype=np.int32)

        self.last_active_step = np.zeros((self.n_rollout_threads, self.num_agents), dtype=np.int32)

    def init_eval_map_variables(self):
        # Action Space
        self.act_dim = self.eval_envs.action_space[0].n
        self.action_shape = get_shape_from_act_space(self.eval_envs.action_space[0])
        if self.use_action_masking:
            self.available_actions = np.ones((self.n_eval_rollout_threads, self.num_agents, self.act_dim), dtype=np.int32)
        else:
            self.available_actions = None

        # Initializing full, merge and local map
        self.macro_action = np.zeros((self.n_eval_rollout_threads, self.num_agents, 2), dtype=np.int32)
        self.agent_pos = np.zeros((self.n_eval_rollout_threads, self.num_agents, 2), dtype=np.int32) # based on the actual map
        self.eval_agent_groups = np.ones((self.n_eval_rollout_threads, self.num_agents, self.num_agents), dtype=np.float32)

        
        self.target_found = np.zeros((self.n_eval_rollout_threads,self.num_agents), dtype=np.int32)
        # self.number_of_rubbles_known = np.zeros((self.n_eval_rollout_threads, self.num_agents), dtype=np.int32)
        
    
    def save_global_model(self, step):
        if len(self.env_infos["sum_merge_explored_reward"]) >= self.all_args.eval_episodes and \
            (np.mean(self.env_infos["sum_merge_explored_reward"]) >= self.best_gobal_reward):
            self.best_gobal_reward = np.mean(self.env_infos["sum_merge_explored_reward"])
            torch.save(self.trainer.policy.actor.state_dict(), str(self.save_dir) + "/global_actor_best.pt")
            torch.save(self.trainer.policy.critic.state_dict(), str(self.save_dir) + "/global_critic_best.pt")
            torch.save(self.trainer.policy.actor_optimizer.state_dict(), str(self.save_dir) + "/global_actor_optimizer_best.pt")
            torch.save(self.trainer.policy.critic_optimizer.state_dict(), str(self.save_dir) + "/global_critic_optimizer_best.pt")  
        torch.save(self.trainer.policy.actor.state_dict(), str(self.save_dir) + "/global_actor_periodic_{}.pt".format(step))
        torch.save(self.trainer.policy.critic.state_dict(), str(self.save_dir) + "/global_critic_periodic_{}.pt".format(step))
        torch.save(self.trainer.policy.actor_optimizer.state_dict(), str(self.save_dir) + "/global_actor_optimizer_periodic_{}.pt".format(step))
        torch.save(self.trainer.policy.critic_optimizer.state_dict(), str(self.save_dir) + "/global_critic_optimizer_periodic_{}.pt".format(step))

    def convert_info(self):
        for k, v in self.env_info.items():
            self.env_infos[k].append(v)
            if k == 'merge_explored_ratio':       
                print('mean merged_explored_ratio:',np.mean(v)*100,'%')
            elif k == 'is_target_found':
                print('mean target_found:',np.mean(v)*100,'%')
            elif k == 'target_rescued':
                print('mean target_rescued:',np.mean(v)*100,'%')
            elif k == 'target_rescued_step':
                print('mean target_rescued_step:',np.nanmean(v))
            elif k == 'avg_total_reward':
                print('mean avg_total_reward:',np.mean(v))
            elif k == 'sum_agent_reward':
                for agent_id in range(self.num_agents):
                    print('mean sum_agent_reward for agent {}:'.format(agent_id),np.mean(v[:,agent_id]))
    
    def convert_eval_info(self):
        for k, v in self.eval_env_info.items():
            self.eval_env_infos[k].append(v)

    def log_env(self, env_infos, total_num_steps):
        for k, v in env_infos.items():
            if len(v) > 0:
                if self.use_wandb:
                    if k == "sum_agent_reward":
                        for agent_id in range(self.num_agents):
                            wandb.log({"agent{}_sum_reward".format(agent_id): np.mean(np.array(v)[0][:,agent_id])}, step=total_num_steps)
                    else:
                        wandb.log({k: np.nanmean(v) if k == "target_rescued_step" or k == "target_found_step" else np.mean(v)}, step=total_num_steps)

                else:
                    self.writter.add_scalars(k, {k: np.nanmean(v) if k == "target_rescued_step" else np.mean(v)}, total_num_steps)

    def log_agent(self, env_infos, total_num_steps):
        for k, v in env_infos.items():
            # print("k and v are: ", k, v)
            if "merge" not in k and "target" not in k:
                for agent_id in range(self.num_agents):
                    agent_k = "agent{}_".format(agent_id) + k
                    if self.use_wandb:
                        wandb.log({agent_k: np.mean(np.array(v)[:,:,agent_id])}, step=total_num_steps)
                    else:
                        print("agent{}_{}: {}".format(agent_id, k, np.mean(np.array(v)[:,:,agent_id])))
                        self.writter.add_scalars(agent_k, {agent_k: np.mean(np.array(v)[:,:,agent_id])}, total_num_steps)

    @torch.no_grad()
    def compute_global_goal(self, step):
        returned_actions = np.zeros((self.n_rollout_threads, self.num_agents, self.action_shape), dtype=np.float32)
        returned_values = np.zeros((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        returned_action_log_probs = np.zeros((self.n_rollout_threads, self.num_agents, self.action_shape), dtype=np.float32)
        returned_rnn_states_actor = np.zeros((self.n_rollout_threads, self.num_agents, self.recurrent_hidden_size), dtype=np.float32)
        returned_rnn_states_critic = np.zeros((self.n_rollout_threads, self.num_agents, self.recurrent_hidden_size), dtype=np.float32)
        
        n_threads = self.n_rollout_threads
        
        # Get short-term goals.
        def get_short_term_goal(self, concat_obs, concat_share_obs,
                                all_rnn_states_actor, all_rnn_states_critic, n_threads, available_actions, active_agents = None):
            value, action, action_log_prob, rnn_states_actor , rnn_states_critic  = self.trainer.policy.get_actions(concat_share_obs,
                                                                                    concat_obs,
                                                                                    np.concatenate(self.buffer.masks[step]),
                                                                                    all_rnn_states_actor,
                                                                                    all_rnn_states_critic,
                                                                                    available_actions,
                                                                                    active_agents)
            
            # [self.envs, agents, dim]
            values = np.array(np.split(_t2n(value), n_threads))
            actions = np.array(np.split(_t2n(action), n_threads))
            action_log_probs = np.array(np.split(_t2n(action_log_prob), n_threads))
            
            if rnn_states_actor is not None:
                rnn_states_actor = np.array(_t2n(rnn_states_actor))
            rnn_states_critic = np.array(_t2n(rnn_states_critic))
            goal = np.array(np.split(_t2n(action), n_threads)).astype(np.int32)
            row = goal//(2*self.action_size+1) - self.action_size
            col = goal%(2*self.action_size+1) - self.action_size
            short_term_goal = np.stack((row, col), axis=-1)
            # short_term_goal = short_term_goal.astype(np.int32) - self.action_size
            return values, actions, action_log_probs, short_term_goal, rnn_states_actor, rnn_states_critic

        self.trainer.prep_rollout()

        # only useful for partial comm
        connected_agent_groups = self.connected_agent_groups
        
        # keys = self.buffer.obs.keys()
        # concat_share_obs = dict.fromkeys(keys)
        # concat_obs = dict.fromkeys(keys)
        concat_share_obs = {}
        concat_obs = {}
        all_available_actions = []
        all_rnn_states_actor = []
        all_rnn_states_critic = []

        obs_shape = get_shape_from_obs_space(self.envs.observation_space[0])
        # print("observation space shape is: ", obs_shape) #would be a dictionary
        
        if self.asynch:
            if self.use_full_comm: # used for centralized and asynchronous training with full communication
                active_threads = self.asynch_control.active_agents_threads()
                for e in range(n_threads):
                    if len(active_threads[e]) <= 1:
                        continue
                    padded_obs = {}
                    for key in obs_shape:
                        padded_obs[key] = np.zeros_like(self.buffer.all_obs[key][step][e])
                        padded_obs[key] = np.expand_dims(padded_obs[key], axis=0)
                    if self.use_action_masking:
                        available_actions = np.ones_like(self.available_actions[0])
                    else:
                        available_actions = None
                    rnn_states_actor = np.zeros_like(self.buffer.rnn_states[0][0])
                    rnn_states_critic = np.zeros_like(self.buffer.rnn_states_critic[0][0])

                    active_cnt = 0
                    inactive_cnt = 0
                    for agent_id in range(self.num_agents):
                        if agent_id in active_threads[e][1:]:
                            for key in obs_shape:
                                padded_obs[key][0,active_cnt] = self.buffer.all_obs[key][step,e,agent_id]

                            if self.use_action_masking:
                                available_actions[active_cnt] = self.available_actions[e][agent_id]
                            rnn_states_actor[active_cnt] = self.buffer.rnn_states[self.asynch_control.cnt[e,agent_id],e,agent_id]
                            rnn_states_critic[active_cnt] = self.buffer.rnn_states_critic[self.asynch_control.cnt[e,agent_id],e,agent_id]
                            # print("agent counters are: ", self.asynch_control.cnt[e,agent_id], e, agent_id)
                            active_cnt += 1
                        else:
                            for key in obs_shape:
                                padded_obs[key][0,-1-inactive_cnt] = self.buffer.all_obs[key][step,e,agent_id]
                            inactive_cnt += 1
                        
                    if not concat_obs:
                        for key in obs_shape:
                            concat_share_obs[key] = padded_obs[key]
                            concat_obs[key] = padded_obs[key]
                    else:
                        for key in obs_shape:
                            concat_share_obs[key] = np.concatenate((concat_share_obs[key],padded_obs[key]))
                            concat_obs[key] = np.concatenate((concat_obs[key],padded_obs[key]))
                    all_available_actions.append(available_actions)
                    all_rnn_states_actor.append(rnn_states_actor)
                    all_rnn_states_critic.append(rnn_states_critic)

                n_stacked_threads = concat_obs['agent_class_identifier'].shape[0]
                if self.use_action_masking:
                    all_available_actions = np.array(all_available_actions)
                else:
                    all_available_actions = None
                all_rnn_states_actor = np.array(all_rnn_states_actor)
                all_rnn_states_critic = np.array(all_rnn_states_critic)
                for key in obs_shape:
                    concat_share_obs[key] = np.concatenate(concat_share_obs[key])
                    concat_obs[key] = np.concatenate(concat_obs[key])
                values, actions, action_log_probs, short_term_goal, rnn_states_actor, rnn_states_critic \
                        = get_short_term_goal(self, concat_obs, concat_share_obs,
                                              all_rnn_states_actor, all_rnn_states_critic, n_stacked_threads, all_available_actions)
                counter = 0 # used to move through the concatenated actions resulting from concatenated obs
                for e in range(n_threads):
                    if len(active_threads[e]) <= 1:
                        continue
                    agent_num = 0
                    for agent_id in active_threads[e][1:]:
                        self.macro_action[e, agent_id] = short_term_goal[counter, agent_num] + self.agent_pos[e, agent_id]
                        self.macro_action[e, agent_id] = self.correct_ma_bounds(self.macro_action[e, agent_id])
                        returned_values[e, agent_id] = values[counter, agent_num]
                        returned_actions[e, agent_id] = actions[counter, agent_num,:]
                        returned_action_log_probs[e, agent_id] = action_log_probs[counter, agent_num,:]
                        if rnn_states_actor is not None:
                            returned_rnn_states_actor[e, agent_id] = rnn_states_actor[counter, agent_num]
                        returned_rnn_states_critic[e, agent_id] = rnn_states_critic[counter, agent_num]
                        agent_num += 1
                    counter += 1
                # print("counter is ", counter)
            elif self.use_partial_comm: # used for asynchronous training with partial communication (distributed training)
                # in each group, puts active agents first, then inactive agents, then zero-paddings
                active_threads = self.asynch_control.active_agents_threads()
                for e in range(n_threads):
                    if len(active_threads[e]) <= 1:
                        continue
                    for group in connected_agent_groups[e]:
                        active_in_group = []
                        inactive_in_group = []
                        for agent_id in group:
                            if agent_id in active_threads[e][1:]:
                                active_in_group.append(agent_id)
                            else:
                                inactive_in_group.append(agent_id)
                        if len(active_in_group) == 0:
                            continue
                        else:
                            padded_obs = {}
                            for key in obs_shape:
                                padded_obs[key] = np.zeros_like(self.buffer.all_obs[key][step][e])
                                padded_obs[key] = np.expand_dims(padded_obs[key], axis=0)
                            if self.use_action_masking:
                                available_actions = np.ones_like(self.available_actions[0])
                            else:
                                available_actions = None
                            rnn_states_actor = np.zeros_like(self.buffer.rnn_states[0][0])
                            rnn_states_critic = np.zeros_like(self.buffer.rnn_states_critic[0][0])
                            agent_num = 0
                            for agent_id in active_in_group:
                                for key in obs_shape:
                                    padded_obs[key][0,agent_num] = self.buffer.all_obs[key][step,e,agent_id]
                                if self.use_action_masking:
                                    available_actions[agent_num] = self.available_actions[e][agent_id]
                                rnn_states_actor[agent_num] = self.buffer.rnn_states[self.asynch_control.cnt[e,agent_id],e,agent_id]
                                rnn_states_critic[agent_num] = self.buffer.rnn_states_critic[self.asynch_control.cnt[e,agent_id],e,agent_id]
                                agent_num += 1
                            for agent_id in inactive_in_group:
                                for key in obs_shape:
                                    padded_obs[key][0,agent_num] = self.buffer.all_obs[key][step,e,agent_id]
                                # available actions and rnn_states are not needed for inactive agents
                                agent_num += 1 

                            if not concat_obs:
                                for key in obs_shape:
                                    concat_share_obs[key] = padded_obs[key]
                                    concat_obs[key] = padded_obs[key]
                            else:
                                for key in obs_shape:
                                    concat_share_obs[key] = np.concatenate((concat_share_obs[key],padded_obs[key]))
                                    concat_obs[key] = np.concatenate((concat_obs[key],padded_obs[key]))
                            all_available_actions.append(available_actions)
                            all_rnn_states_actor.append(rnn_states_actor)
                            all_rnn_states_critic.append(rnn_states_critic)

                n_stacked_threads = concat_obs['agent_class_identifier'].shape[0] # total groups with at least one active agent
                if self.use_action_masking:
                    all_available_actions = np.array(all_available_actions)
                else:
                    all_available_actions = None
                all_rnn_states_actor = np.array(all_rnn_states_actor)
                all_rnn_states_critic = np.array(all_rnn_states_critic)

                for key in obs_shape:
                    concat_share_obs[key] = np.concatenate(concat_share_obs[key])
                    concat_obs[key] = np.concatenate(concat_obs[key])
                values, actions, action_log_probs, short_term_goal, rnn_states_actor, rnn_states_critic \
                        = get_short_term_goal(self, concat_obs, concat_share_obs,
                                              all_rnn_states_actor, all_rnn_states_critic, n_stacked_threads, all_available_actions)
                counter = 0 # used to move through the concatenated actions resulting from concatenated obs
                for e in range(n_threads):
                    if len(active_threads[e]) <= 1:
                        continue
                    for group in connected_agent_groups[e]:
                        agent_num = 0
                        any_active = False
                        for agent_id in group:
                            if agent_id in active_threads[e][1:]:
                                any_active = True
                                self.macro_action[e, agent_id] = short_term_goal[counter, agent_num] + self.agent_pos[e, agent_id]
                                self.macro_action[e, agent_id] = self.correct_ma_bounds(self.macro_action[e, agent_id])
                                returned_values[e, agent_id] = values[counter, agent_num]
                                returned_actions[e, agent_id] = actions[counter, agent_num,:]
                                returned_action_log_probs[e, agent_id] = action_log_probs[counter, agent_num,:]
                                returned_rnn_states_actor[e, agent_id] = rnn_states_actor[counter, agent_num]
                                returned_rnn_states_critic[e, agent_id] = rnn_states_critic[counter, agent_num]
                                agent_num += 1
                        if any_active:
                            counter += 1
                # print("counter is ", counter)
        else:
            if self.use_full_comm: # used for centralized and synchronous training with full communication
                for key in obs_shape:
                    concat_share_obs[key] = np.concatenate(self.buffer.share_obs[key][step])
                for key in obs_shape:
                    concat_obs[key] = np.concatenate(self.buffer.obs[key][step])
                
                returned_values, returned_actions, returned_action_log_probs, short_term_goal, returned_rnn_states_actor, returned_rnn_states_critic \
                        = get_short_term_goal(self, concat_obs, concat_share_obs,
                                              self.buffer.rnn_states[step], self.buffer.rnn_states_critic[step], n_threads, self.available_actions)
                short_term_goal = short_term_goal.squeeze() + self.agent_pos
                self.macro_action = self.correct_ma_bounds(short_term_goal)
            elif self.use_partial_comm: # used for centralized and synchronous training with partial communication
                for e in range(self.n_rollout_threads):
                    for group in connected_agent_groups[e]:
                        L = len(group)
                        padded_obs = {}
                        for key in obs_shape:
                            padded_obs[key] = np.zeros_like(self.buffer.obs[key][step][e])
                            padded_obs[key] = np.expand_dims(padded_obs[key], axis=0)
                        if self.use_action_masking:
                            available_actions = np.ones_like(self.available_actions[0])
                        else:
                            available_actions = None
                        rnn_states_actor = np.zeros_like(self.buffer.rnn_states[0][0])
                        rnn_states_critic = np.zeros_like(self.buffer.rnn_states_critic[0][0])
                        agent_num = 0
                        for agent_id in group:
                            for key in obs_shape:
                                padded_obs[key][0,agent_num] = self.buffer.obs[key][step,e,agent_id]
                            if self.use_action_masking:
                                available_actions[agent_num] = self.available_actions[e][agent_id]
                            rnn_states_actor[agent_num] = self.buffer.rnn_states[step,e,agent_id]
                            rnn_states_critic[agent_num] = self.buffer.rnn_states_critic[step,e,agent_id]
                            agent_num += 1
                        if not concat_obs:
                            for key in obs_shape:
                                concat_share_obs[key] = padded_obs[key]
                                concat_obs[key] = padded_obs[key]
                        else:
                            for key in obs_shape:
                                concat_share_obs[key] = np.concatenate((concat_share_obs[key],padded_obs[key]))
                                concat_obs[key] = np.concatenate((concat_obs[key],padded_obs[key]))
                        all_available_actions.append(available_actions)
                        all_rnn_states_actor.append(rnn_states_actor)
                        all_rnn_states_critic.append(rnn_states_critic)
                n_stacked_threads = concat_obs['agent_class_identifier'].shape[0]
                if self.use_action_masking:
                    all_available_actions = np.array(all_available_actions)
                else:
                    all_available_actions = None
                all_rnn_states_actor = np.array(all_rnn_states_actor)
                all_rnn_states_critic = np.array(all_rnn_states_critic)

                for key in obs_shape:
                    concat_share_obs[key] = np.concatenate(concat_share_obs[key])
                    concat_obs[key] = np.concatenate(concat_obs[key])
                values, actions, action_log_probs, short_term_goal, rnn_states_actor, rnn_states_critic \
                        = get_short_term_goal(self, concat_obs, concat_share_obs,
                                              all_rnn_states_actor, all_rnn_states_critic,  n_stacked_threads, all_available_actions)
                
                counter = 0 # used to move through the concatenated actions resulting from concatenated obs
                for e in range(self.n_rollout_threads):
                    for group in connected_agent_groups[e]:
                        agent_num = 0
                        for agent_id in group:
                            self.macro_action[e, agent_id] = short_term_goal[counter,agent_num] + self.agent_pos[e, agent_id]
                            self.macro_action[e, agent_id] = self.correct_ma_bounds(self.macro_action[e, agent_id])
                            returned_values[e, agent_id] = values[counter,agent_num]
                            returned_actions[e, agent_id] = actions[counter,agent_num,:]
                            returned_action_log_probs[e, agent_id] = action_log_probs[counter,agent_num,:]
                            returned_rnn_states_actor[e, agent_id] = rnn_states_actor[counter, agent_num]
                            returned_rnn_states_critic[e, agent_id] = rnn_states_critic[counter, agent_num]
                            agent_num += 1
                        counter += 1

        return returned_values, returned_actions, returned_action_log_probs, returned_rnn_states_actor, returned_rnn_states_critic
    
    def eval_compute_global_goal(self, step, infos, use_ft):
        if self.use_render:
            n_threads = self.n_rollout_threads
        elif self.use_eval:
            n_threads = self.n_eval_rollout_threads
        connected_agent_groups = []
        for e in range(n_threads):
            connected_agent_groups.append(infos[e]['connected_agent_groups'])
        returned_rnn_states = np.zeros((n_threads, self.num_agents, self.recurrent_hidden_size), dtype=np.float32)
        
        # function to return short-term goals and rnn_states (optional) using the trained policy
        def get_short_term_goal(self, concat_obs, concat_share_obs, rnn_states, n_threads, available_actions = None):
            if self.use_render:
                action, rnn_states = self.trainer.policy.act(
                    concat_share_obs,
                    concat_obs,
                    np.concatenate(self.masks),
                    rnn_states,
                    available_actions = available_actions,
                    deterministic=True
                )
                # self.rnn_states = np.array(np.split(_t2n(rnn_states), n_threads))
            elif self.use_eval:
                action, rnn_states = self.trainer.policy.act(
                    concat_share_obs,
                    concat_obs,
                    np.concatenate(self.eval_masks),
                    rnn_states,
                    available_actions = available_actions,
                    deterministic=True
                )
                # self.eval_rnn_states = np.array(np.split(_t2n(rnn_states), n_threads))
            
            rnn_states = np.array(_t2n(rnn_states))
            # short_term_goals = np.array(np.split(_t2n(action), n_threads))
            
            # short_term_goal = short_term_goals.astype(np.int32) - self.action_size

            goal = np.array(np.split(_t2n(action), n_threads)).astype(np.int32)
            row = goal//(2*self.action_size+1) - self.action_size
            col = goal%(2*self.action_size+1) - self.action_size
            short_term_goal = np.stack((row, col), axis=-1)
            return short_term_goal, rnn_states
    
        if use_ft:
            if self.use_render:
                self.ft_short_term_goals = self.envs.ft_get_short_term_goals(self.all_args, mode=self.all_args.algorithm_name[3:])
            elif self.use_eval:
                self.ft_short_term_goals = self.eval_envs.ft_get_short_term_goals(self.all_args, mode=self.all_args.algorithm_name[3:])
            # Used to render for ft methods.
            if (not self.asynch) or self.all_args.use_time:
                self.macro_action = np.array([
                    [
                        # (x, y) ---> (y, x) in minigrid
                        (goal[1] - self.agent_view_size, goal[0] - self.agent_view_size)
                        for goal in env_goals
                    ]
                    for env_goals in self.ft_short_term_goals
                ])
            else:
                short_term_goals = [
                    [
                        # (x, y) ---> (y, x) in minigrid
                        (goal[1] - self.agent_view_size, goal[0] - self.agent_view_size)
                        for goal in env_goals
                    ]
                    for env_goals in self.ft_short_term_goals
                ]
                if not hasattr(self, 'macro_action'):
                        self.macro_action = np.zeros((self.n_rollout_threads, self.num_agents, 2), dtype=int)
                self.macro_action = (short_term_goals * self.asynch_control.active.reshape(self.n_rollout_threads, self.num_agents, 1)).astype(int) \
                    + (self.macro_action * (1-self.asynch_control.active.reshape(self.n_rollout_threads, self.num_agents, 1))).astype(int)
            
            self.macro_action[self.macro_action>=self.map_size]=self.map_size-1
            self.macro_action[self.macro_action<0]=0
            # print("macro_action is: ", self.macro_action)
        
        else:
            self.trainer.prep_rollout()

            concat_obs = {}
            concat_share_obs = {}
            all_available_actions = []
            all_rnn_states = [] # the critic rnn states do not matter during evaluation
            if self.use_render:
                obs = self.obs
                old_rnn_states = self.rnn_states
            elif self.use_eval:
                obs = self.eval_obs
                old_rnn_states = self.eval_rnn_states
            if self.use_partial_comm: 
                # form concat_obs such that the observations in each connected_agent_groups are put there first, and the rest are zero-paddings
                if self.asynch: # Asynchronous and partial communication
                    # in each group, puts active agents first, then inactive agents, then zero-paddings
                    active_threads = self.asynch_control.active_agents_threads()
                    for e in range(n_threads):
                        if len(active_threads[e]) <= 1:
                                continue
                        for group in connected_agent_groups[e]:
                            
                            padded_obs = {}
                            for key in obs.keys():
                                padded_obs[key] = np.zeros_like(obs[key][e])
                                padded_obs[key] = np.expand_dims(padded_obs[key], axis=0)
                            available_actions = np.zeros_like(self.available_actions[0])
                            rnn_states = np.zeros_like(old_rnn_states[0,0])

                            active_in_group = []
                            inactive_in_group = []
                            for agent_id in group:
                                if agent_id in active_threads[e][1:]:
                                    active_in_group.append(agent_id)
                                else:
                                    inactive_in_group.append(agent_id)
                            if len(active_in_group) == 0:
                                continue
                            else:
                                agent_num = 0
                                for agent_id in active_in_group:
                                    for key in obs.keys():
                                        padded_obs[key][0,agent_num] = obs[key][e,agent_id]
                                    available_actions[agent_num] = self.available_actions[e][agent_id]
                                    rnn_states[agent_num] = old_rnn_states[self.asynch_control.cnt[e,agent_id],e,agent_id]
                                    agent_num += 1
                                for agent_id in inactive_in_group:
                                    for key in obs.keys():
                                        padded_obs[key][0,agent_num] = obs[key][e,agent_id]
                                    agent_num += 1 
                            if not concat_obs:
                                for key in obs.keys():
                                    concat_share_obs[key] = padded_obs[key]
                                    concat_obs[key] = padded_obs[key]
                            else:
                                for key in obs.keys():
                                    concat_share_obs[key] = np.concatenate((concat_share_obs[key],padded_obs[key]))
                                    concat_obs[key] = np.concatenate((concat_obs[key],padded_obs[key]))
                            all_available_actions.append(available_actions)
                            all_rnn_states.append(rnn_states)

                    n_stacked_threads = concat_obs['agent_class_identifier'].shape[0]
                    for key in obs.keys():
                        concat_share_obs[key] = np.concatenate(concat_share_obs[key])
                        concat_obs[key] = np.concatenate(concat_obs[key])
                    all_available_actions = np.array(all_available_actions)
                    all_rnn_states = np.array(all_rnn_states)
                    short_term_goal, rnn_states = get_short_term_goal(self, concat_obs, concat_share_obs, all_rnn_states, n_stacked_threads, all_available_actions)
                    
                    counter = 0 # used to move through the concatenated actions resulting from concatenated obs
                    for e in range(n_threads):
                        if len(active_threads[e]) <= 1:
                            continue
                        for group in connected_agent_groups[e]:
                            active_in_group = []
                            inactive_in_group = []
                            for agent_id in group:
                                if agent_id in active_threads[e][1:]:
                                    active_in_group.append(agent_id)
                                else:
                                    inactive_in_group.append(agent_id)
                            if len(active_in_group) == 0:
                                continue
                            else:
                                group_goals = short_term_goal[counter,:,:]
                                agent_num = 0
                                for agent_id in active_in_group: # only active agents get new short_term_goal
                                    self.macro_action[e, agent_id] = group_goals[agent_num,:] + self.agent_pos[e, agent_id]
                                    self.macro_action[e, agent_id] = self.correct_ma_bounds(self.macro_action[e, agent_id])
                                    if rnn_states is not None:
                                        returned_rnn_states[e, agent_id] = rnn_states[counter, agent_num]
                                    agent_num += 1
                                counter += 1
                else:
                    for e in range(n_threads):
                        for group in connected_agent_groups[e]:
                            L = len(group)
                            padded_obs = {}
                            for key in obs.keys():
                                padded_obs[key] = np.zeros_like(obs[key][e])
                                padded_obs[key] = np.expand_dims(padded_obs[key], axis=0)
                            available_actions = np.zeros_like(self.available_actions[0])
                            rnn_states = np.zeros_like(old_rnn_states[0,0])
                            agent_num = 0
                            for agent_id in group:
                                for key in obs.keys():
                                    padded_obs[key][0,agent_num] = obs[key][e,agent_id]
                                available_actions[agent_num] = self.available_actions[e][agent_id]
                                rnn_states[agent_num] = old_rnn_states[step,e,agent_id]
                                agent_num += 1
            
                            if not concat_obs:
                                for key in obs.keys():
                                    concat_share_obs[key] = padded_obs[key]
                                    concat_obs[key] = padded_obs[key]
                            else:
                                for key in obs.keys():
                                    concat_share_obs[key] = np.concatenate((concat_share_obs[key],padded_obs[key]))
                                    concat_obs[key] = np.concatenate((concat_obs[key],padded_obs[key]))
                            all_available_actions.append(available_actions)
                            all_rnn_states.append(rnn_states)
                    n_stacked_threads = concat_obs['agent_class_identifier'].shape[0]
                    for key in obs.keys():
                        concat_share_obs[key] = np.concatenate(concat_share_obs[key])
                        concat_obs[key] = np.concatenate(concat_obs[key])
                    all_available_actions = np.array(all_available_actions)
                    all_rnn_states = np.array(all_rnn_states)
                    short_term_goal, rnn_states = get_short_term_goal(self, concat_obs, concat_share_obs, all_rnn_states, n_stacked_threads, all_available_actions)
                    counter = 0 # used to move through the concatenated groups
                    for e in range(n_threads):
                        for group in connected_agent_groups[e]:
                            group_goals = short_term_goal[counter,:,:]
                            agent_num = 0
                            for agent_id in group:
                                self.macro_action[e, agent_id] = group_goals[agent_num,:] + self.agent_pos[e, agent_id]
                                self.macro_action[e, agent_id] = self.correct_ma_bounds(self.macro_action[e, agent_id])
                                if rnn_states is not None:
                                    returned_rnn_states[e, agent_id] = rnn_states[counter, agent_num]
                                agent_num += 1
                            counter += 1

            elif self.use_full_comm:
                if self.asynch: #asynchronous and full communication
                    active_threads = self.asynch_control.active_agents_threads()
                    for e in range(n_threads):
                        if len(active_threads[e]) <= 1:
                            continue
                        padded_obs = {}
                        for key in obs.keys():
                            padded_obs[key] = np.zeros_like(obs[key][e])
                            padded_obs[key] = np.expand_dims(padded_obs[key], axis=0)
                        if self.use_action_masking:
                            available_actions = np.zeros_like(self.available_actions[0])
                        else:
                            available_actions = None
                        rnn_states = np.zeros_like(old_rnn_states[0,0])

                        active_cnt = 0
                        inactive_cnt = 0
                        for agent_id in range(self.num_agents):
                            if agent_id in active_threads[e][1:]:
                                for key in obs.keys():
                                    padded_obs[key][0,active_cnt] = obs[key][e,agent_id]
                                
                                if self.use_action_masking:
                                    available_actions[active_cnt] = self.available_actions[e][agent_id]
                                rnn_states[active_cnt] = old_rnn_states[self.asynch_control.cnt[e,agent_id],e,agent_id]
                                active_cnt += 1
                            else:
                                for key in obs.keys():
                                    padded_obs[key][0,-1-inactive_cnt] = obs[key][e,agent_id]
                                # padded_obs['global_agent_map'][0,agent_num,:,:,:] = obs['global_agent_map'][e,agent_id,:,:,:]
                                # padded_obs['agent_class_identifier'][0,agent_num,:] = obs['agent_class_identifier'][e,agent_id,:]
                                inactive_cnt += 1
                            
                        if not concat_obs:
                            for key in obs.keys():
                                concat_share_obs[key] = padded_obs[key]
                                concat_obs[key] = padded_obs[key]
                        else:
                            for key in obs.keys():
                                concat_share_obs[key] = np.concatenate((concat_share_obs[key],padded_obs[key]))
                                concat_obs[key] = np.concatenate((concat_obs[key],padded_obs[key]))
                        all_available_actions.append(available_actions)
                        all_rnn_states.append(rnn_states)
                    n_stacked_threads = concat_obs['agent_class_identifier'].shape[0]
                    if self.use_action_masking:
                        all_available_actions = np.array(all_available_actions)
                    else:
                        all_available_actions = None
                    all_rnn_states = np.array(all_rnn_states)

                    for key in obs.keys():
                        concat_share_obs[key] = np.concatenate(concat_share_obs[key])
                        concat_obs[key] = np.concatenate(concat_obs[key])
                    short_term_goal, rnn_states = get_short_term_goal(self, concat_obs, concat_share_obs, all_rnn_states, n_stacked_threads, all_available_actions)
                    counter = 0 # used to move through the concatenated actions resulting from concatenated obs
                    for e in range(n_threads):
                        if len(active_threads[e]) <= 1:
                            continue
                        agent_num = 0
                        for agent_id in active_threads[e][1:]:
                            self.macro_action[e, agent_id] = short_term_goal[counter, agent_num] + self.agent_pos[e, agent_id]
                            self.macro_action[e, agent_id] = self.correct_ma_bounds(self.macro_action[e, agent_id])
                            if rnn_states is not None:
                                returned_rnn_states[e, agent_id] = rnn_states[counter, agent_num]
                            agent_num += 1
                        counter += 1
                    
                else: #synchronous and full communication
                    for key in obs.keys():
                        concat_obs[key] = np.concatenate(obs[key])
                        concat_share_obs[key] = np.concatenate(obs[key])
                    available_actions = np.array(self.available_actions)
                    short_term_goal, returned_rnn_states = get_short_term_goal(self, concat_obs, concat_share_obs, old_rnn_states[step], n_threads, available_actions)
                    short_term_goal = short_term_goal.squeeze() + self.agent_pos
                    self.macro_action = self.correct_ma_bounds(short_term_goal)
        return returned_rnn_states
        
    @torch.no_grad()
    def compute(self, next_values):
        self.trainer.prep_rollout()
        
        """
        # This is migrated to the run() function
        # we have to find each agent (e,a) last active timstep, and concatenate the obs from that step over all the agents        
        
        concat_obs = {}
        concat_share_obs = {}
        
        if self.asynch:
            for e in range(self.n_rollout_threads):
                for a in range(self.num_agents):
                    last_index = self.buffer.active_steps[self.buffer.update_step[e,a], e, a]
                    if not concat_obs:
                        for key in self.buffer.obs.keys():
                            concat_share_obs[key] = self.buffer.all_obs[key][last_index, e]
                            concat_obs[key] = self.buffer.all_obs[key][last_index, e]
                    else:
                        for key in self.buffer.obs.keys():
                            concat_share_obs[key] = np.concatenate((concat_share_obs[key],self.buffer.all_obs[key][last_index, e]))
                            concat_obs[key] = np.concatenate((concat_obs[key],self.buffer.all_obs[key][last_index, e]))
            # The resulting number of threads would be n_rollout_threads*num_agents
            for key in self.buffer.share_obs.keys(): 
                concat_share_obs[key] = np.squeeze(concat_share_obs[key])
                concat_obs[key] = np.squeeze(concat_obs[key])
                concat_share_obs[key] = np.concatenate(concat_share_obs[key])
                concat_obs[key] = np.concatenate(concat_obs[key])
            # print("shape of concat_obs['global_agent_map'] is now: ", concat_obs['global_agent_map'].shape)
            next_values = self.trainer.policy.get_values(concat_share_obs,
                                                        concat_obs,
                                                        np.concatenate(
                                                            self.buffer.rnn_states_critic[-1]),
                                                        np.concatenate(self.buffer.masks[-1]))
            next_values = np.array(np.split(_t2n(next_values), self.n_rollout_threads*self.num_agents))
            # print("shape of next_values is: ", next_values.shape)
            counter = 0
            actual_next_values = np.zeros((self.n_rollout_threads, self.num_agents), dtype=np.float32)
            for e in range(self.n_rollout_threads):
                for a in range(self.num_agents):
                    actual_next_values[e, a] = next_values[counter][a]
                    counter += 1
            # print("actual_next_values are: ", actual_next_values)
            
            self.buffer.async_compute_returns(actual_next_values, self.trainer.value_normalizer)
        else:
            #concat is done to merge threads and agents
            for key in self.buffer.obs.keys():
                concat_obs[key] = np.concatenate(self.buffer.obs[key][-1])
            for key in self.buffer.share_obs.keys():
                concat_share_obs[key] = np.concatenate(self.buffer.share_obs[key][-1])
            # print("shape of concat_obs['global_agent_map'] is: ", concat_obs['global_agent_map'].shape)

            
            # The value of observations when making the final decision of the episode
            next_values = self.trainer.policy.get_values(concat_share_obs,
                                                        concat_obs,
                                                        np.concatenate(
                                                            self.buffer.rnn_states_critic[-1]),
                                                        np.concatenate(self.buffer.masks[-1]))
            next_values = np.array(np.split(_t2n(next_values), self.n_rollout_threads)) 
            
            # print("next values are: ", next_values)
            # print("shape of next_values is: ", next_values.shape)
            self.buffer.compute_returns(next_values, self.trainer.value_normalizer) """
        if self.asynch:
            self.buffer.async_compute_returns(next_values, self.trainer.value_normalizer)
        else:
            self.buffer.compute_returns(next_values, self.trainer.value_normalizer)

    def insert(self, data, step, active_agents=None):
        dict_obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic, agent_groups, available_actions = data
        dones_env = np.all(dones, axis=-1)
         
        # rnn_states[dones_env == True] = np.zeros(
        #     ((dones_env == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
        # rnn_states_critic[dones_env == True] = np.zeros(((dones_env == True).sum(
        # ), self.num_agents, *self.buffer.rnn_states_critic.shape[3:]), dtype=np.float32)
        
        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones_env == True] = np.zeros(
            ((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)


        if active_agents is None:
            obs = self._convert(dict_obs, infos, step)
        else:
            obs = self._convert(dict_obs, infos, step, active_agents)
        self.obs = obs
        share_obs = obs.copy()


        if active_agents is None:
            self.buffer.insert(share_obs, obs, rnn_states, rnn_states_critic, actions, action_log_probs, values, rewards, masks, agent_groups, available_actions=available_actions)
        else:
            self.buffer.async_insert(share_obs, obs, rnn_states, rnn_states_critic, actions, action_log_probs, values, rewards, masks, agent_groups, available_actions=available_actions, active_agents=active_agents)

    @torch.no_grad()
    def eval(self):
        
        if self.asynch:
            def generate_random_period(min_t,max_t):
                return 0
            self.asynch_control = AsynchControl(num_envs=self.n_eval_rollout_threads, num_agents=self.num_agents,
                                                limit=self.episode_length, random_fn=generate_random_period, min_wait=2, max_wait=5, rest_time = 10)
        eval_envs = self.eval_envs
        self.eval_env_infos = defaultdict(list)
        use_ft = self.all_args.algorithm_name[:2] == "ft"

        # Visualization of the results
        mission_completion_data = []

        for episode in range(self.all_args.eval_episodes):
            ic(episode)
            self.init_eval_env_info()
            self.init_eval_map_variables()
            self.eval_rnn_states = np.zeros((self.max_steps+1, self.n_eval_rollout_threads, self.num_agents, self.recurrent_hidden_size),dtype=np.float32)
            self.eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
            reset_choose = np.ones(self.n_eval_rollout_threads) == 1.0
            eval_dict_obs, eval_infos = eval_envs.reset(reset_choose)
            for e in range(self.n_eval_rollout_threads):
                self.agent_pos[e] = eval_infos[e]['agent_pos'] #initial agent positions
                self.agent_types_list[e] = np.array(eval_infos[e]['agent_types_list'])
            
            is_last_step = np.full((self.n_eval_rollout_threads, self.num_agents), False)

            # one-hot agent_class_identifier
            self.agent_class_identifier = np.zeros((self.n_eval_rollout_threads, self.num_agents, self.n_agent_types), dtype=np.float32)
            for e in range(self.n_eval_rollout_threads):
                for agent_id in range(self.num_agents):
                    for agent_type in range(self.n_agent_types):
                        if self.agent_types_list[e, agent_id] == agent_type:
                            self.agent_class_identifier[e, agent_id, agent_type] = 1

            if self.asynch:
                self.asynch_control.reset()
            
            if self.use_action_masking:
                available_actions = self.get_available_actions(self.eval_envs)
                self.available_actions[0] = available_actions[0]
            else:
                available_actions = None
                self.available_actions = None


            eval_obs = self._convert(eval_dict_obs, eval_infos, 0)
            self.eval_obs = eval_obs
            eval_rnn_states = self.eval_compute_global_goal(0, eval_infos, use_ft)
            if not self.asynch:
                self.eval_rnn_states[0] = eval_rnn_states
            else:
                self.eval_rnn_states[0] = eval_rnn_states

            eval_full_episode_rewards = np.zeros((self.n_eval_rollout_threads, self.num_agents), dtype=np.float32)
            
            for step in range(self.max_steps):

                local_step = step % self.local_step_num

                eval_actions_env = eval_envs.get_short_term_action(self.macro_action)
                eval_actions_env = np.array(eval_actions_env)

                # number_of_rubbles_known = self.number_of_rubbles_known.copy()

                eval_dict_obs, eval_rewards, eval_dones, eval_infos = eval_envs.step(eval_actions_env)
    
                # Adding information to their respective dictionary keys.
                for e in range(self.n_eval_rollout_threads):
                    for key in eval_infos[e].keys():
                        if key == 'merge_explored_ratio':
                            if np.all(eval_dones[e]):
                                self.eval_env_info['eval_'+key][e] = eval_infos[e][key]
                        elif 'eval_'+key in self.eval_env_info_keys:
                            self.eval_env_info['eval_'+key][e] = eval_infos[e][key]
                        if key in self.target_info_keys:
                            if key == 'target_found':
                                self.eval_env_info[key][e] = int(eval_infos[e][key].any())
                            else:
                                self.eval_env_info[key][e] = eval_infos[e][key]
                        # if key in self.rubble_info_keys:
                        #     self.eval_env_info[key][e] = max(self.eval_env_info[key][e],eval_infos[e][key])
                    # self.number_of_rubbles_known[e] = eval_infos[e]['agent_number_of_known_rubbles']
                    self.agent_pos[e] = eval_infos[e]['agent_pos']
                    self.agent_alive[e] = eval_infos[e]['agent_alive'] #agent alive status

                if self.asynch:
                    self.asynch_control.step()
                    for e in range(self.n_eval_rollout_threads):
                        for a in range(self.num_agents):
                            if self.agent_alive[e][a] == 0:
                                self.asynch_control.standby[e, a] = 0
                                self.asynch_control.active[e, a] = 0
                                is_last_step[e,a] = True
                                continue
                            if not self.asynch_control.active[e, a] and not self.asynch_control.standby[e, a]:
                                # Checks if agent has reached its short term goal / stopped and puts it on standby if so
                                #TODO: If the agent has reached its ultimate objective, instead of putting it on standby, it should be completely deactivated
                                # Under the assumption of full communication in training, this is added to ensure the final MA of all agents finish at the same time
                                if self.eval_env_info["target_rescued"][e] and not is_last_step[e,a]:
                                    is_last_step[e,a] = True
                                    # self.asynch_control.standby[e, a] = 1
                                    self.asynch_control.activate(e, a)
                                
                                elif eval_actions_env[e, a] == 3: #stop action
                                    self.asynch_control.standby[e, a] = 1
                                    # self.asynch_control.activate(e, a)
                                    # eval_rewards[e, a, 0] -= 5
                                
                                elif eval_actions_env[e, a] == 4: #infeasible action
                                    self.asynch_control.standby[e, a] = 1
                                    # self.asynch_control.activate(e, a)
                                    # eval_rewards[e, a, 0] -= 5
                                    
                                elif self.target_found[e, a] == 0:
                                    if eval_infos[e]['target_found'][a] == 1:
                                        self.target_found[e, a] = 1
                                        self.asynch_control.standby[e, a] = 1
                                        # self.asynch_control.activate(e, a)

                                # # if self.agent_types_list[a] == 2 and number_of_rubbles_known[e,a] != self.number_of_rubbles_known[e,a] \
                                # elif number_of_rubbles_known[e,a] != self.number_of_rubbles_known[e,a]:
                                #     self.asynch_control.standby[e, a] = 1
                                #     # self.asynch_control.activate(e, a)
                    
                            
                                         
                    if np.any(self.asynch_control.standby) and np.any(self.asynch_control.active): #Activates on-standby agents based on communication model
                        for thread in self.asynch_control.active_agents_threads():
                            if len(thread) > 1:
                                if self.use_partial_comm:
                                    connected_agents = eval_infos[thread[0]]['connected_agent_groups']
                                    # print("connected agent groups are" , connected_agents)
                                    for agent_id in thread[1:]:
                                        for group in connected_agents:
                                            if agent_id in group:
                                                for i in group:
                                                    if self.asynch_control.standby[thread[0], i]:
                                                    # if self.asynch_control.standby[thread[0], i] and self.asynch_control.wait[thread[0], i] <= generate_random_period(2, 4):
                                                        self.asynch_control.activate(thread[0], i)
                                elif self.use_full_comm:
                                    for i in range(self.num_agents):
                                        if self.asynch_control.standby[thread[0], i] and self.asynch_control.wait[thread[0], i] <= generate_random_period(2, 4):
                                            self.asynch_control.activate(thread[0], i)
                else:
                    for e in range(self.n_rollout_threads):
                        for a in range(self.num_agents):
                            if self.agent_alive[e][a] == 0:
                                is_last_step[e,a] = True
                                continue
                            if self.eval_env_info["target_rescued"][e] and not is_last_step[e,a]:
                                is_last_step[e,a] = True
                                # rewards[e, a, 0] = 0
                
                eval_full_episode_rewards += eval_rewards[:,:,0]
                
                if (not self.asynch and local_step == self.local_step_num - 1) or (self.asynch and np.any(self.asynch_control.active)):
                    eval_obs = self._convert(eval_dict_obs, eval_infos, step)
                    self.eval_obs = eval_obs
                    eval_dones_env = np.all(eval_dones, axis=-1)
                    # self.eval_rnn_states[eval_dones_env == True] = np.zeros(
                    #     ((eval_dones_env == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
                    self.eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
                    self.eval_masks[eval_dones_env == True] = np.zeros(
                        ((eval_dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)
                    
                    if self.use_action_masking:
                        available_actions = self.get_available_actions(self.eval_envs)
                        self.available_actions[0] = available_actions[0]
                    else:
                        available_actions = None
                        self.available_actions = None

                    eval_rnn_states = self.eval_compute_global_goal(step, eval_infos, use_ft)
                    if not self.asynch:
                        self.eval_rnn_states[step] = eval_rnn_states
                    else:
                        for (e, a, s) in self.asynch_control.active_agents():
                            self.eval_rnn_states[s, e, a] = eval_rnn_states[e, a]

                if is_last_step.all():
                    break    
            
            completion_time_data = np.where(np.isnan(self.eval_env_info['target_rescued_step']), -1, self.eval_env_info['target_rescued_step'])
            print("completion time data is: ", completion_time_data)
            mission_completion_data.extend(completion_time_data)
            print("episode reward is:", eval_full_episode_rewards)
            self.convert_eval_info()
        print("completeion info is ", mission_completion_data)
        
        print("eval average stayed at target rate is: " +
                str(np.mean(self.eval_env_infos['target_rescued'])))
        print("eval average stayed at target step is: " +
                str(np.nanmean(self.eval_env_infos['target_rescued_step'])))
        print("eval average merge explored ratio is: " +
                str(np.mean(self.eval_env_infos['eval_merge_explored_ratio'])))

        # Save data to file
        np.save('Task3_catmip.npy', mission_completion_data)

        # Visualize the results
        success_counts = np.zeros(self.max_steps + 1)
        for value in mission_completion_data:
            if value != -1:
                success_counts[int(value)] += 1
        cumulative_success_counts = np.cumsum(success_counts)
        total_episodes = len(mission_completion_data)
        success_rate = cumulative_success_counts / total_episodes
        # Plot the cumulative success probability
        plt.figure(figsize=(10, 6))
        plt.plot(range(self.max_steps + 1), success_rate, label='Cumulative Success Probability')
        plt.xlabel('Time Step')
        plt.ylabel('Probability')
        plt.title('Probability of Mission Success Before Time Step t')
        plt.legend()
        plt.show()
        
        

    @torch.no_grad()
    def render(self):
        if self.asynch:
            def generate_random_period(min_t,max_t):
                # return np.random.randint(min_t, max_t)
                return 0
            self.asynch_control = AsynchControl(num_envs=self.n_rollout_threads, num_agents=self.num_agents,
                                                limit=self.episode_length, random_fn=generate_random_period, min_wait=2, max_wait=5, rest_time = 20)
        
        envs = self.envs
        # Init env infos.
        self.eval_infos = defaultdict(list)
        use_ft = self.all_args.algorithm_name[:2] == "ft"
        all_frames = []
        all_local_frames = []

        for episode in range(self.all_args.render_episodes):
            ic(episode)
            self.init_env_info()
            self.init_map_variables()
            reset_choose = np.ones(self.n_rollout_threads) == 1.0
            dict_obs, infos = envs.reset(reset_choose)
            for e in range(self.n_rollout_threads):
                self.agent_pos[e] = infos[e]['agent_pos'] #initial agent positions
                self.agent_types_list[e] = np.array(infos[e]['agent_types_list'])
            print("Agent types are: ", self.agent_types_list)
            
            self.rnn_states = np.zeros((self.max_steps+1, self.n_rollout_threads, self.num_agents, self.recurrent_hidden_size),dtype=np.float32)
            self.masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
            is_last_step = np.full((self.n_rollout_threads, self.num_agents), False)
            
            # one-hot agent_class_identifier
            self.agent_class_identifier = np.zeros((self.n_rollout_threads, self.num_agents, self.n_agent_types), dtype=np.int32)
            for e in range(self.n_rollout_threads):
                for agent_id in range(self.num_agents):
                    for agent_type in range(self.n_agent_types):
                        if self.agent_types_list[e, agent_id] == agent_type:
                            self.agent_class_identifier[e, agent_id, agent_type] = 1

                
            if self.asynch:
                self.asynch_control.reset()

            if self.use_action_masking:
                available_actions = self.get_available_actions(self.envs)
                self.available_actions[0] = available_actions[0]
            else:
                available_actions = None
                self.available_actions = None


            obs = self._convert(dict_obs, infos, 0)
            self.obs = obs
            rnn_states = self.eval_compute_global_goal(0, infos, use_ft)
            if not self.asynch:
                self.rnn_states[0] = rnn_states
            else:
                self.rnn_states[0] = rnn_states
            period_rewards = np.zeros((self.n_rollout_threads, self.num_agents, 1))
            
            # if self.use_render: #and episode == 0:
            #     if self.all_args.save_gifs:
            #         # if self.env_info["target_rescued"][e] == 1:
            #         #     pass
            #         # else:
            #         image, local_image = envs.render('rgb_array', self.macro_action)[0]
            #         all_frames.append(image)
            #         all_local_frames.append(local_image)
            #         calc_end = time.time()
            #         elapsed = calc_end - calc_start
            #         if elapsed < self.all_args.ifi:
            #             time.sleep(self.all_args.ifi - elapsed)
            #     else:
            #         # print(self.macro_action)
            #         envs.render('human', self.macro_action)
            # time.sleep(5)

            full_episode_rewards = np.zeros((self.n_rollout_threads, self.num_agents), dtype=np.float32)
            
            for step in range(self.max_steps):
                calc_start = time.time()
                local_step = step % self.local_step_num                    

                if use_ft:
                    actions_env = envs.get_short_term_action(self.macro_action)
                    # actions_env = envs.ft_get_short_term_actions(self.macro_action,self.all_args.astar_cost_mode,self.all_args.astar_utility_radius)
                else:
                    actions_env = envs.get_short_term_action(self.macro_action)
                actions_env = np.array(actions_env)

                dict_obs, rewards, dones, infos = envs.step(actions_env)
                # Adding information to their respective dictionary keys.
                for e in range(self.n_rollout_threads):
                    for key in self.sum_env_info_keys:
                        if key in infos[e].keys():
                            self.env_info['sum_{}'.format(key)][e] += np.array(infos[e][key])
                    for key in self.equal_env_info_keys:
                        if key == 'merge_explored_ratio':
                            if np.all(dones[e]):
                                self.env_info[key][e] = infos[e][key]
                        if key == 'agent_explored_ratio':
                            if np.all(dones[e]):
                                self.env_info[key][e] = infos[e][key]
                        elif key in infos[e].keys():
                            if key == 'explored_ratio_step':
                                for agent_id in range(self.num_agents):
                                    agent_k = "agent{}_{}".format(agent_id, key)
                                    if agent_k in infos[e].keys():
                                        self.env_info[key][e][agent_id] = infos[e][agent_k]
                            else:
                                self.env_info[key][e] = infos[e][key]
                    for key in self.target_info_keys:
                        if key == 'target_found':
                            self.env_info[key][e] = int(infos[e][key].any())
                        elif infos[e][key]:
                            self.env_info[key][e] = infos[e][key]
                    
                    # self.number_of_rubbles_known[e] = infos[e]['agent_number_of_known_rubbles']
                    self.agent_pos[e] = infos[e]['agent_pos']
                    self.agent_alive[e] = infos[e]['agent_alive'] #agent alive status
                    
                    # print("target_rescued step is ", self.env_info["target_rescued_step"][e])
                    # if dones[e].all() and step != self.max_steps - 1:
                    #     self.env_info["target_rescued"][e] = 1
                        # print("env {} is done at step {}".format(e, step))
                    

                if self.asynch:
                    self.asynch_control.step()
                    for e in range(self.n_rollout_threads):
                        for a in range(self.num_agents):
                            if self.agent_alive[e][a] == 0:
                                self.asynch_control.standby[e, a] = 0
                                self.asynch_control.active[e, a] = 0
                                is_last_step[e,a] = True
                                continue
                            if not self.asynch_control.active[e, a] and not self.asynch_control.standby[e, a]:
                                # Checks if agent has reached its short term goal / stopped and puts it on standby if so
                                #TODO: If the agent has reached its ultimate objective, instead of putting it on standby, it should be completely deactivated

                                # Under the assumption of full communication in training, this is added to ensure the final MA of all agents finish at the same time
                                if self.env_info["target_rescued"][e] and not is_last_step[e,a]:
                                    is_last_step[e,a] = True
                                    # self.asynch_control.standby[e, a] = 1
                                    self.asynch_control.activate(e, a)

                                elif actions_env[e, a] == 3: #stop action
                                    self.asynch_control.standby[e, a] = 1
                                    # self.asynch_control.activate(e, a)
                                    # rewards[e, a, 0] -= 5
                                
                                elif actions_env[e, a] == 4: #infeasible action
                                    self.asynch_control.standby[e, a] = 1
                                    # self.asynch_control.activate(e, a)
                                    # rewards[e, a, 0] -= 5
                                    
                                elif self.target_found[e, a] == 0:
                                    if infos[e]['target_found'][a] == 1:
                                        self.target_found[e, a] = 1
                                        self.asynch_control.standby[e, a] = 1
                                        # self.asynch_control.activate(e, a)

                                # # if self.agent_types_list[a] == 2 and number_of_rubbles_known[e,a] != self.number_of_rubbles_known[e,a] \
                                # elif number_of_rubbles_known[e,a] != self.number_of_rubbles_known[e,a]:
                                #     self.asynch_control.standby[e, a] = 1
                                #     # self.asynch_control.activate(e, a)
                                

                    if np.any(self.asynch_control.standby) and np.any(self.asynch_control.active): #Activates on-standby agents based on communication model
                        for thread in self.asynch_control.active_agents_threads():
                            if len(thread) > 1:
                                if self.use_partial_comm:
                                    connected_agents = infos[thread[0]]['connected_agent_groups']
                                    # print("connected agent groups are" , connected_agents)
                                    for agent_id in thread[1:]:
                                        for group in connected_agents:
                                            if agent_id in group:
                                                for i in group:
                                                    if self.asynch_control.standby[thread[0], i]:
                                                    # if self.asynch_control.standby[thread[0], i] and self.asynch_control.wait[thread[0], i] <= generate_random_period(2, 4):
                                                        self.asynch_control.activate(thread[0], i)
                                elif self.use_full_comm:
                                    for i in range(self.num_agents):
                                        if self.asynch_control.standby[thread[0], i] and self.asynch_control.wait[thread[0], i] <= generate_random_period(2, 4):
                                            self.asynch_control.activate(thread[0], i)                  
                else:
                    for e in range(self.n_rollout_threads):
                        for a in range(self.num_agents):
                            if self.agent_alive[e][a] == 0:
                                is_last_step[e,a] = True
                                continue
                            if self.env_info["target_rescued"][e] and not is_last_step[e,a]:
                                is_last_step[e,a] = True
                                # rewards[e, a, 0] = 0
                
                full_episode_rewards += rewards[:,:,0]
                period_rewards += rewards
                
                if (not self.asynch and local_step == self.local_step_num - 1) or (self.asynch and np.any(self.asynch_control.active)):
                    if self.asynch:
                        for e, a, s in self.asynch_control.active_agents():
                            period_rewards[e, a, 0] = 0
                    else:
                        # ic(period_rewards) #for monitoring the rewards
                        period_rewards = np.zeros((self.n_rollout_threads, self.num_agents, 1))
                    
                    obs = self._convert(dict_obs, infos, step)
                    self.obs = obs #to be used in eval_compute_global_goal
                    
                    dones_env = np.all(dones, axis=-1)
                    # self.rnn_states[dones_env == True] = np.zeros(
                    #     ((dones_env == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
                    self.masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
                    self.masks[dones_env == True] = np.zeros(
                        ((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)
                    
                    
                    if self.use_action_masking:
                        available_actions = self.get_available_actions(self.envs)
                        self.available_actions[0] = available_actions[0]
                    else:
                        available_actions = None
                        self.available_actions = None
                    rnn_states = self.eval_compute_global_goal(step, infos, use_ft)

                    if not self.asynch:
                        self.rnn_states[step] = rnn_states
                    else:
                        for (e, a, s) in self.asynch_control.active_agents():
                            self.rnn_states[s, e, a] = rnn_states[e, a]
                            # print("e a s are: ", e, a, s)
                    # print("rnn states are: ", self.rnn_states[:5,:,:]) # use this for debugging
                    
                    
                
                
                # print(self.env_info['target_rescued_step'])
                if self.use_render: #and episode == 0:
                    if self.all_args.save_gifs:
                        # if self.env_info["target_rescued"][e] == 1:
                        #     pass
                        # else:
                        image, local_image = envs.render('rgb_array', self.macro_action)[0]
                        all_frames.append(image)
                        all_local_frames.append(local_image)
                        calc_end = time.time()
                        elapsed = calc_end - calc_start
                        if elapsed < self.all_args.ifi:
                            time.sleep(self.all_args.ifi - elapsed)
                    else:
                        # print(self.macro_action)
                        envs.render('human', self.macro_action)
                        # pass
                
                # if step > 12:
                #     time.sleep(5) # pause for debugging for 30 secs
                if is_last_step.all():
                    break
            
            print("episode reward is: ", full_episode_rewards)
            self.convert_info()
            total_num_steps = (episode + 1) * self.max_steps * self.n_rollout_threads
            if not self.use_render :
                self.log_env(self.env_infos, total_num_steps)
                self.log_agent(self.env_infos, total_num_steps)
            
        for k, v in self.env_infos.items():
            print("eval average {}: {}".format(k, np.nanmean(v) if k == 'merge_explored_ratio_step' or k == "merge_explored_ratio_step_0.98"else np.mean(v)))

        if self.all_args.save_gifs:
            ic("rendering....")
            imageio.mimsave(str(self.gif_dir) + '/merge.gif',
                            all_frames, duration=self.all_args.ifi)
            imageio.mimsave(str(self.gif_dir) + '/local.gif',
                            all_local_frames, duration=self.all_args.ifi)
            ic("done")

    