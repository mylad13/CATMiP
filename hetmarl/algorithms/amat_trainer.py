import numpy as np
import torch
import torch.nn as nn
from hetmarl.utils.util import get_gard_norm, huber_loss, mse_loss
from hetmarl.utils.valuenorm import ValueNorm
from hetmarl.algorithms.utils.util import check
from hetmarl.algorithms.utils.util import get_connected_agents

class AMATTrainer:
    """
    Trainer class for MAT to update policies.
    :param args: (argparse.Namespace) arguments containing relevant model, policy, and env information.
    :param policy: (R_MAPPO_Policy) policy to update.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """
    def __init__(self,
                 args,
                 policy,
                 num_agents,
                 device=torch.device("cpu")):

        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.policy = policy
        self.num_agents = num_agents

        self.clip_param = args.clip_param
        self.ppo_epoch = args.ppo_epoch
        self.num_mini_batch = args.num_mini_batch
        self.data_chunk_length = args.data_chunk_length
        self.value_loss_coef = args.value_loss_coef
        self.entropy_coef = args.entropy_coef
        self.max_grad_norm = args.max_grad_norm       
        self.huber_delta = args.huber_delta

        self._use_recurrent_policy = args.use_recurrent_policy
        self._use_naive_recurrent = args.use_naive_recurrent_policy
        self._use_max_grad_norm = args.use_max_grad_norm
        self._use_clipped_value_loss = args.use_clipped_value_loss
        self._use_huber_loss = args.use_huber_loss
        self._use_valuenorm = args.use_valuenorm
        self._use_value_active_masks = args.use_value_active_masks
        self._use_policy_active_masks = args.use_policy_active_masks
        self.dec_actor = args.dec_actor
        self.asynch = args.asynch
        self.use_full_comm = args.use_full_comm
        self.use_partial_comm = args.use_partial_comm
        
        if self._use_valuenorm:
            self.value_normalizer = ValueNorm(1, device=self.device)
        else:
            self.value_normalizer = None

    def cal_value_loss(self, values, value_preds_batch, return_batch, active_masks_batch):
        """
        Calculate value function loss.
        :param values: (torch.Tensor) value function predictions.
        :param value_preds_batch: (torch.Tensor) "old" value  predictions from data batch (used for value clip loss)
        :param return_batch: (torch.Tensor) reward to go returns.
        :param active_masks_batch: (torch.Tensor) denotes if agent is active or dead at a given timesep.

        :return value_loss: (torch.Tensor) value function loss.
        """
        # print("values are: ", values)
        # print("old values are: ", value_preds_batch)
        value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(-self.clip_param,
                                                                                    self.clip_param)

        if self._use_valuenorm:
            self.value_normalizer.update(return_batch)
            error_clipped = self.value_normalizer.normalize(return_batch) - value_pred_clipped
            error_original = self.value_normalizer.normalize(return_batch) - values
        else:
            error_clipped = return_batch - value_pred_clipped
            error_original = return_batch - values

        if self._use_huber_loss:
            value_loss_clipped = huber_loss(error_clipped, self.huber_delta)
            value_loss_original = huber_loss(error_original, self.huber_delta)
        else:
            value_loss_clipped = mse_loss(error_clipped)
            value_loss_original = mse_loss(error_original)

        if self._use_clipped_value_loss:
            value_loss = torch.max(value_loss_original, value_loss_clipped)
        else:
            value_loss = value_loss_original

        # if self._use_value_active_masks and not self.dec_actor:
        if self._use_value_active_masks:
            value_loss = (value_loss * active_masks_batch).sum() / active_masks_batch.sum()
        else:
            value_loss = value_loss.mean()

        return value_loss

    def ppo_update(self, sample): #TODO: make sure active_masks are used correctly to mask out info after envs are done
        """
        Update actor and critic networks.
        :param sample: (Tuple) contains data batch with which to update networks.
        :update_actor: (bool) whether to update actor network.

        :return value_loss: (torch.Tensor) value function loss.
        :return critic_grad_norm: (torch.Tensor) gradient norm from critic up9date.
        ;return policy_loss: (torch.Tensor) actor(policy) loss value.
        :return dist_entropy: (torch.Tensor) action entropies.
        :return actor_grad_norm: (torch.Tensor) gradient norm from actor update.
        :return imp_weights: (torch.Tensor) importance sampling weights.
        """
        share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, \
        value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, \
        adv_targ, available_actions_batch, agent_groups_batch = sample
        # print("batch size in sync ppo is: ", return_batch.shape[0])
        
        if self.use_partial_comm: #unused
            batch_size = agent_groups_batch.shape[0] #batch_size is episode_length*n_threads
            obs = {}
            share_obs = {}
            concat_obs = {}
            concat_actions = []
            concat_masks = []
            concat_active_masks = []
            concat_available_actions = []
            concat_rnn_states = []
            concat_rnn_states_critic = []
            for key in obs_batch.keys():
                obs[key] = obs_batch[key].reshape(-1, self.num_agents, *obs_batch[key].shape[1:])
                share_obs[key] = share_obs_batch[key].reshape(-1, self.num_agents, *share_obs_batch[key].shape[1:])
            actions = actions_batch.reshape(-1, self.num_agents, *actions_batch.shape[1:])
            masks = masks_batch.reshape(-1, self.num_agents, *masks_batch.shape[1:])
            active_masks = active_masks_batch.reshape(-1, self.num_agents, *active_masks_batch.shape[1:])
            available_actions = available_actions_batch.reshape(-1, self.num_agents, *available_actions_batch.shape[1:])
            rnn_states = rnn_states_batch.reshape(-1, self.num_agents, *rnn_states_batch.shape[1:])
            rnn_states_critic = rnn_states_critic_batch.reshape(-1, self.num_agents, *rnn_states_critic_batch.shape[1:])
            
            connected_agent_groups = []
            for i in range(batch_size):
                connected_agent_groups.append(get_connected_agents(agent_groups_batch[i]))
                for group in connected_agent_groups[i]:
                    padded_obs = {}
                    for key in obs.keys():
                        padded_obs[key] = np.zeros_like(obs[key][i])
                        padded_obs[key] = np.expand_dims(padded_obs[key], axis=0)
                    padded_actions = np.zeros_like(actions[i])
                    padded_masks = np.zeros_like(masks[i])
                    padded_active_masks = np.zeros_like(active_masks[i])
                    padded_available_actions = np.zeros_like(available_actions[i])
                    padded_rnn_states = np.zeros_like(rnn_states[i])
                    padded_rnn_states_critic = np.zeros_like(rnn_states_critic[i])
                    agent_num = 0
                    for agent_id in group:
                        for key in obs.keys():
                            padded_obs[key][0,agent_num] = obs[key][i][agent_id]
                        padded_actions[agent_num] = actions[i][agent_id]
                        padded_masks[agent_num] = masks[i][agent_id]
                        padded_active_masks[agent_num] = active_masks[i][agent_id]
                        padded_available_actions[agent_num] = available_actions[i][agent_id]
                        padded_rnn_states[agent_num] = rnn_states[i][agent_id]
                        padded_rnn_states_critic[agent_num] = rnn_states_critic[i][agent_id]
                        agent_num += 1
                    if not concat_obs:
                        for key in obs.keys():
                            concat_obs[key] = padded_obs[key]
                    else:
                        for key in obs.keys():
                            concat_obs[key] = np.concatenate((concat_obs[key], padded_obs[key]))
                    concat_actions.append(padded_actions)
                    concat_masks.append(padded_masks)
                    concat_active_masks.append(padded_active_masks)
                    concat_available_actions.append(padded_available_actions)
                    concat_rnn_states.append(padded_rnn_states)
                    concat_rnn_states_critic.append(padded_rnn_states_critic)
            for key in obs_batch.keys():
                concat_obs[key] = concat_obs[key].reshape(-1, *obs_batch[key].shape[1:])
            concat_actions = np.array(concat_actions).reshape(-1, *actions_batch.shape[1:])
            concat_masks = np.array(concat_masks).reshape(-1, *masks_batch.shape[1:])
            concat_active_masks = np.array(concat_active_masks).reshape(-1, *active_masks_batch.shape[1:])
            concat_share_obs = concat_obs.copy()
            concat_available_actions = np.array(concat_available_actions).reshape(-1, *available_actions_batch.shape[1:])
            concat_rnn_states = np.array(concat_rnn_states).reshape(-1, *rnn_states_batch.shape[1:])
            concat_rnn_states_critic = np.array(concat_rnn_states_critic).reshape(-1, *rnn_states_critic_batch.shape[1:])


            old_action_log_probs_batch = check(old_action_log_probs_batch).to(**self.tpdv)
            adv_targ = check(adv_targ).to(**self.tpdv)
            value_preds_batch = check(value_preds_batch).to(**self.tpdv)
            return_batch = check(return_batch).to(**self.tpdv)
            active_masks_batch = check(active_masks_batch).to(**self.tpdv)
            concat_active_masks = check(concat_active_masks).to(**self.tpdv)
            concat_available_actions = check(concat_available_actions).to(**self.tpdv)
            concat_rnn_states = check(concat_rnn_states).to(**self.tpdv)
            concat_rnn_states_critic = check(concat_rnn_states_critic).to(**self.tpdv)

            pre_values, pre_action_log_probs, dist_entropy = self.policy.evaluate_actions(concat_share_obs,
                                                                              concat_obs, 
                                                                              rnn_states_batch, 
                                                                              rnn_states_critic_batch, 
                                                                              concat_actions, 
                                                                              concat_masks, 
                                                                              available_actions_batch,
                                                                              concat_active_masks)
            pre_values = pre_values.view(-1, self.num_agents, 1)
            pre_action_log_probs = pre_action_log_probs.view(-1, self.num_agents, 1)
            values = np.zeros((batch_size, self.num_agents, 1))
            action_log_probs = np.zeros((batch_size, self.num_agents, 1))
            values = check(values).to(**self.tpdv)
            action_log_probs = check(action_log_probs).to(**self.tpdv)
            counter = 0
            for i in range(batch_size):
                for group in connected_agent_groups[i]:
                    agent_num = 0
                    for agent_id in group:
                        values[i][agent_id] = pre_values[counter,agent_num]
                        action_log_probs[i][agent_id] = pre_action_log_probs[counter,agent_num]
                        agent_num += 1
                    counter += 1
            values = values.view(-1, 1)
            action_log_probs = action_log_probs.view(-1, 1)
        elif self.use_full_comm:
            old_action_log_probs_batch = check(old_action_log_probs_batch).to(**self.tpdv)
            adv_targ = check(adv_targ).to(**self.tpdv)
            value_preds_batch = check(value_preds_batch).to(**self.tpdv)
            return_batch = check(return_batch).to(**self.tpdv)
            active_masks_batch = check(active_masks_batch).to(**self.tpdv)

            # Reshaped to do in a single forward pass for all steps
            values, action_log_probs, dist_entropy = self.policy.evaluate_actions(share_obs_batch,
                                                                                obs_batch, 
                                                                                rnn_states_batch, 
                                                                                rnn_states_critic_batch, 
                                                                                actions_batch, 
                                                                                masks_batch, 
                                                                                available_actions_batch,
                                                                                active_masks_batch)

        # actor update
        imp_weights = torch.exp(action_log_probs - old_action_log_probs_batch)

        surr1 = imp_weights * adv_targ
        surr2 = torch.clamp(imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ

        if self._use_policy_active_masks:
            policy_loss = (-torch.sum(torch.min(surr1, surr2),
                                      dim=-1,
                                      keepdim=True) * active_masks_batch).sum() / active_masks_batch.sum()
        else:
            policy_loss = -torch.sum(torch.min(surr1, surr2), dim=-1, keepdim=True).mean()

        # critic update
        value_loss = self.cal_value_loss(values, value_preds_batch, return_batch, active_masks_batch)

        loss = policy_loss - dist_entropy * self.entropy_coef + value_loss * self.value_loss_coef

        self.policy.optimizer.zero_grad()
        loss.backward()

        if self._use_max_grad_norm:
            grad_norm = nn.utils.clip_grad_norm_(self.policy.transformer.parameters(), self.max_grad_norm)
        else:
            grad_norm = get_gard_norm(self.policy.transformer.parameters())
        
        self.policy.optimizer.step()

        return value_loss, grad_norm, policy_loss, dist_entropy, grad_norm, imp_weights
    
    def async_ppo_update(self, sample):
        """
        Update actor and critic networks.
        :param sample: (Tuple) contains data batch with which to update networks.
        :update_actor: (bool) whether to update actor network.

        :return value_loss: (torch.Tensor) value function loss.
        :return critic_grad_norm: (torch.Tensor) gradient norm from critic up9date.
        ;return policy_loss: (torch.Tensor) actor(policy) loss value.
        :return dist_entropy: (torch.Tensor) action entropies.
        :return actor_grad_norm: (torch.Tensor) gradient norm from actor update.
        :return imp_weights: (torch.Tensor) importance sampling weights.
        """
        share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, \
        value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, \
        adv_targ, available_actions_batch, active_agents_batch, agent_groups_batch = sample
        

        if self.use_partial_comm:
            # batch_size = active_agents_batch.shape[0] 
            batch_size = agent_groups_batch.shape[0] 
            # batch_size = return_batch.shape[0] 
            print("batch size in async ppo with partial communication is: ", batch_size)
            # print("agent group batch is: ", agent_groups_batch)
            obs = {}
            share_obs = {}
            concat_obs = {}
            concat_actions = []
            concat_masks = []
            concat_active_masks = []
            concat_available_actions = []
            concat_rnn_states = []
            concat_rnn_states_critic = []
            for key in obs_batch.keys():
                obs[key] = obs_batch[key].reshape(-1, self.num_agents, *obs_batch[key].shape[1:])
                share_obs[key] = share_obs_batch[key].reshape(-1, self.num_agents, *share_obs_batch[key].shape[1:])
            actions = actions_batch.reshape(-1, self.num_agents, *actions_batch.shape[1:])
            masks = masks_batch.reshape(-1, self.num_agents, *masks_batch.shape[1:])
            active_masks = active_masks_batch.reshape(-1, self.num_agents, *active_masks_batch.shape[1:])
            active_agents = active_agents_batch.reshape(-1, self.num_agents)
            available_actions = available_actions_batch.reshape(-1, self.num_agents, *available_actions_batch.shape[1:])
            rnn_states = rnn_states_batch.reshape(-1, self.num_agents, *rnn_states_batch.shape[1:])
            rnn_states_critic = rnn_states_critic_batch.reshape(-1, self.num_agents, *rnn_states_critic_batch.shape[1:])

            connected_agent_groups = []
            for i in range(batch_size):
                connected_agent_groups.append(get_connected_agents(agent_groups_batch[i]))
                for group in connected_agent_groups[i]:
                    active_in_group = []
                    inactive_in_group = []
                    for agent_id in group:
                        if active_agents[i,agent_id]:
                            active_in_group.append(agent_id)
                        else:
                            inactive_in_group.append(agent_id)
                    if len(active_in_group) == 0:
                        continue
                    else:
                        padded_obs = {}
                        for key in obs.keys():
                            padded_obs[key] = np.zeros_like(obs[key][i])
                            padded_obs[key] = np.expand_dims(padded_obs[key], axis=0)
                        padded_actions = np.zeros_like(actions[i])
                        padded_masks = np.zeros_like(masks[i])
                        padded_active_masks = np.zeros_like(active_masks[i])
                        padded_available_actions = np.zeros_like(available_actions[i])
                        padded_rnn_states = np.zeros_like(rnn_states[i])
                        padded_rnn_states_critic = np.zeros_like(rnn_states_critic[i])
                        agent_num = 0
                        for agent_id in active_in_group:
                            for key in obs.keys():
                                padded_obs[key][0,agent_num] = obs[key][i][agent_id]
                            padded_actions[agent_num] = actions[i][agent_id]
                            padded_masks[agent_num] = masks[i][agent_id]
                            padded_active_masks[agent_num] = active_masks[i][agent_id]
                            padded_available_actions[agent_num] = available_actions[i][agent_id]
                            padded_rnn_states[agent_num] = rnn_states[i][agent_id]
                            padded_rnn_states_critic[agent_num] = rnn_states_critic[i][agent_id]
                            agent_num += 1
                        for agent_id in inactive_in_group:
                            for key in obs.keys():
                                padded_obs[key][0,agent_num] = obs[key][i][agent_id]
                            padded_actions[agent_num] = actions[i][agent_id]
                            padded_masks[agent_num] = masks[i][agent_id]
                            padded_active_masks[agent_num] = active_masks[i][agent_id]
                            padded_available_actions[agent_num] = available_actions[i][agent_id]
                            padded_rnn_states[agent_num] = rnn_states[i][agent_id]
                            padded_rnn_states_critic[agent_num] = rnn_states_critic[i][agent_id]
                            agent_num += 1
                    if not concat_obs:
                        for key in obs.keys():
                            concat_obs[key] = padded_obs[key]
                    else:
                        for key in obs.keys():
                            concat_obs[key] = np.concatenate((concat_obs[key], padded_obs[key]))
                    concat_actions.append(padded_actions)
                    concat_masks.append(padded_masks)
                    concat_active_masks.append(padded_active_masks)
                    concat_available_actions.append(padded_available_actions)
                    concat_rnn_states.append(padded_rnn_states)
                    concat_rnn_states_critic.append(padded_rnn_states_critic)
            for key in obs_batch.keys():
                concat_obs[key] = concat_obs[key].reshape(-1, *obs_batch[key].shape[1:])
            concat_actions = np.array(concat_actions).reshape(-1, *actions_batch.shape[1:])
            concat_masks = np.array(concat_masks).reshape(-1, *masks_batch.shape[1:])
            concat_active_masks = np.array(concat_active_masks).reshape(-1, *active_masks_batch.shape[1:])
            concat_available_actions = np.array(concat_available_actions).reshape(-1, *available_actions_batch.shape[1:])
            concat_rnn_states = np.array(concat_rnn_states).reshape(-1, *rnn_states_batch.shape[1:])
            concat_rnn_states_critic = np.array(concat_rnn_states_critic).reshape(-1, *rnn_states_critic_batch.shape[1:])

            concat_share_obs = concat_obs.copy()

            old_action_log_probs_batch = check(old_action_log_probs_batch).to(**self.tpdv)
            adv_targ = check(adv_targ).to(**self.tpdv)
            value_preds_batch = check(value_preds_batch).to(**self.tpdv)
            return_batch = check(return_batch).to(**self.tpdv)
            active_masks_batch = check(active_masks_batch).to(**self.tpdv)
            concat_active_masks = check(concat_active_masks).to(**self.tpdv)
            concat_available_actions = check(concat_available_actions).to(**self.tpdv)
            concat_rnn_states = check(concat_rnn_states).to(**self.tpdv)
            concat_rnn_states_critic = check(concat_rnn_states_critic).to(**self.tpdv)

            pre_values, pre_action_log_probs, dist_entropy = self.policy.evaluate_actions(concat_share_obs,
                                                                              concat_obs, 
                                                                              concat_rnn_states, 
                                                                              concat_rnn_states_critic, 
                                                                              concat_actions, 
                                                                              concat_masks, 
                                                                              concat_available_actions,
                                                                              concat_active_masks)
            pre_values = pre_values.view(-1, self.num_agents, 1)
            pre_action_log_probs = pre_action_log_probs.view(-1, self.num_agents, 1) #(-1, self.num_agents, 2) for multidiscrete action space with two action heads
            values = np.zeros((batch_size, self.num_agents, 1))
            action_log_probs = np.zeros((batch_size, self.num_agents, 1))
            values = check(values).to(**self.tpdv)
            action_log_probs = check(action_log_probs).to(**self.tpdv)
            counter = 0
            for i in range(batch_size):
                for group in connected_agent_groups[i]:
                    agent_num = 0
                    any_active = False
                    for agent_id in group:
                        if active_agents[i,agent_id]:
                            any_active = True
                            values[i][agent_id] = pre_values[counter,agent_num]
                            action_log_probs[i][agent_id] = pre_action_log_probs[counter,agent_num]
                            agent_num += 1
                    if any_active:
                        counter += 1
            values = values.view(-1, 1)
            action_log_probs = action_log_probs.view(-1, 1)
            pass
        elif self.use_full_comm:
            batch_size = return_batch.shape[0] #batch size is episode_length*n_threads*n_agents ?
            print("batch size in async ppo is: ", batch_size)
            old_action_log_probs_batch = check(old_action_log_probs_batch).to(**self.tpdv)
            adv_targ = check(adv_targ).to(**self.tpdv)
            value_preds_batch = check(value_preds_batch).to(**self.tpdv)
            return_batch = check(return_batch).to(**self.tpdv)
            active_masks_batch = check(active_masks_batch).to(**self.tpdv)   

            # Reshape to do in a single forward pass for all steps
            values, action_log_probs, dist_entropy = self.policy.evaluate_actions(share_obs_batch,
                                                                                obs_batch, 
                                                                                rnn_states_batch, 
                                                                                rnn_states_critic_batch, 
                                                                                actions_batch, 
                                                                                masks_batch, 
                                                                                available_actions_batch,
                                                                                active_masks_batch,
                                                                                active_agents_batch)
            #only keep values that belong to active agents
            stacked_old_action_log_probs = []
            stacked_adv_targ = []
            stacked_value_preds_batch = []
            stacked_return_batch = []
            stacked_active_masks_batch = []
            stacked_values = []
            stacked_action_log_probs = []
            for i in range(batch_size):
                if active_agents_batch[i]==True:
                    stacked_old_action_log_probs.append(old_action_log_probs_batch[i])
                    stacked_adv_targ.append(adv_targ[i])
                    stacked_value_preds_batch.append(value_preds_batch[i])
                    stacked_return_batch.append(return_batch[i])
                    stacked_active_masks_batch.append(active_masks_batch[i])
                    stacked_values.append(values[i])
                    stacked_action_log_probs.append(action_log_probs[i])
            old_action_log_probs_batch = torch.stack(stacked_old_action_log_probs)
            adv_targ = torch.stack(stacked_adv_targ)
            value_preds_batch = torch.stack(stacked_value_preds_batch)
            return_batch = torch.stack(stacked_return_batch)
            active_masks_batch = torch.stack(stacked_active_masks_batch)
            values = torch.stack(stacked_values)
            action_log_probs = torch.stack(stacked_action_log_probs)

        # actor update
        imp_weights = torch.exp(action_log_probs - old_action_log_probs_batch)

        surr1 = imp_weights * adv_targ
        surr2 = torch.clamp(imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ

        if self._use_policy_active_masks:
            policy_loss = (-torch.sum(torch.min(surr1, surr2),
                                      dim=-1,
                                      keepdim=True) * active_masks_batch).sum() / active_masks_batch.sum()
        else:
            policy_loss = -torch.sum(torch.min(surr1, surr2), dim=-1, keepdim=True).mean()

        # critic update
        value_loss = self.cal_value_loss(values, value_preds_batch, return_batch, active_masks_batch)

        loss = policy_loss - dist_entropy * self.entropy_coef + value_loss * self.value_loss_coef

        self.policy.optimizer.zero_grad()
        loss.backward()

        if self._use_max_grad_norm:
            grad_norm = nn.utils.clip_grad_norm_(self.policy.transformer.parameters(), self.max_grad_norm)
        else:
            grad_norm = get_gard_norm(self.policy.transformer.parameters())

        self.policy.optimizer.step()

        return value_loss, grad_norm, policy_loss, dist_entropy, grad_norm, imp_weights

    def train(self, buffer):
        """
        Perform a training update using minibatch GD.
        :param buffer: (SharedReplayBuffer) buffer containing training data.
        :param update_actor: (bool) whether to update actor network.

        :return train_info: (dict) contains information regarding training update (e.g. loss, grad norms, etc).
        """
        advantages_copy = buffer.advantages.copy()
        advantages_copy[buffer.active_masks[:-1] == 0.0] = np.nan
        if self.asynch:
            advantages_copy[buffer.update_step_masks[:-1] == 0.0] = np.nan
        mean_advantages = np.nanmean(advantages_copy)
        std_advantages = np.nanstd(advantages_copy)
        advantages = (buffer.advantages - mean_advantages) / (std_advantages + 1e-5)
        

        train_info = {}

        train_info['value_loss'] = 0
        train_info['policy_loss'] = 0
        train_info['dist_entropy'] = 0
        train_info['actor_grad_norm'] = 0
        train_info['critic_grad_norm'] = 0
        train_info['ratio'] = 0

        for _ in range(self.ppo_epoch):
            if self.asynch:
                data_generator = buffer.async_feed_forward_generator_transformer(advantages, self.num_mini_batch)
                for sample in data_generator:

                    value_loss, critic_grad_norm, policy_loss, dist_entropy, actor_grad_norm, imp_weights \
                        = self.async_ppo_update(sample)

                    train_info['value_loss'] += value_loss.item()
                    train_info['policy_loss'] += policy_loss.item()
                    train_info['dist_entropy'] += dist_entropy.item()
                    train_info['actor_grad_norm'] += actor_grad_norm
                    train_info['critic_grad_norm'] += critic_grad_norm
                    train_info['ratio'] += imp_weights.mean()

            else:
                data_generator = buffer.feed_forward_generator_transformer(advantages, self.num_mini_batch)

                for sample in data_generator:

                    value_loss, critic_grad_norm, policy_loss, dist_entropy, actor_grad_norm, imp_weights \
                        = self.ppo_update(sample)

                    train_info['value_loss'] += value_loss.item()
                    train_info['policy_loss'] += policy_loss.item()
                    train_info['dist_entropy'] += dist_entropy.item()
                    train_info['actor_grad_norm'] += actor_grad_norm
                    train_info['critic_grad_norm'] += critic_grad_norm
                    train_info['ratio'] += imp_weights.mean()

        num_updates = self.ppo_epoch * self.num_mini_batch

        for k in train_info.keys():
            train_info[k] /= num_updates
 
        return train_info

    def prep_training(self):
        self.policy.train()

    def prep_rollout(self):
        self.policy.eval()
