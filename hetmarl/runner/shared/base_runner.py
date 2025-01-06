
import time
import wandb
import os
import numpy as np
from itertools import chain
import torch
from tensorboardX import SummaryWriter
from hetmarl.utils.shared_buffer import SharedReplayBuffer
from hetmarl.utils.util import update_linear_schedule
import icecream as ic

def _t2n(x):
    return x.detach().cpu().numpy()


class Runner(object):
    def __init__(self, config):

        self.all_args = config['all_args']
        self.envs = config['envs']
        self.eval_envs = config['eval_envs']
        self.device = config['device']
        self.num_agents = config['num_agents']
        if config.__contains__("render_envs"):
            self.render_envs = config['render_envs']


    
        # parameters
        self.env_name = self.all_args.env_name
        self.algorithm_name = self.all_args.algorithm_name
        self.experiment_name = self.all_args.experiment_name
        self.use_centralized_V = self.all_args.use_centralized_V
        self.num_env_steps = self.all_args.num_env_steps
        self.episode_length = self.all_args.episode_length
        self.n_rollout_threads = self.all_args.n_rollout_threads
        self.n_eval_rollout_threads = self.all_args.n_eval_rollout_threads
        self.n_render_rollout_threads = self.all_args.n_render_rollout_threads
        self.use_linear_lr_decay = self.all_args.use_linear_lr_decay
        self.hidden_size = self.all_args.hidden_size
        self.use_wandb = self.all_args.use_wandb
        self.use_single_network = self.all_args.use_single_network
        self.use_render = self.all_args.use_render
        self.recurrent_N = self.all_args.recurrent_N

        # agent types
        self.n_agent_types = self.all_args.n_agent_types
        self.agent_types_list = np.zeros((self.n_rollout_threads, self.num_agents), dtype=int)

        # action space
        self.action_size = self.all_args.action_size


        # interval
        self.save_interval = self.all_args.save_interval
        self.use_eval = self.all_args.use_eval
        self.eval_interval = self.all_args.eval_interval
        self.log_interval = self.all_args.log_interval

        # dir
        self.model_dir = self.all_args.model_dir

        if self.use_render:
            import imageio
            self.run_dir = config["run_dir"]
            self.gif_dir = str(self.run_dir / 'gifs')
            if not os.path.exists(self.gif_dir):
                os.makedirs(self.gif_dir)
        else:
            if self.use_wandb:
                self.save_dir = str(wandb.run.dir)
                self.run_dir = str(wandb.run.dir)
            else:
                self.run_dir = config["run_dir"]
                self.log_dir = str(self.run_dir / 'logs')
                if not os.path.exists(self.log_dir):
                    os.makedirs(self.log_dir)
                self.writter = SummaryWriter(self.log_dir)
                self.save_dir = str(self.run_dir / 'models')
                if not os.path.exists(self.save_dir):
                    os.makedirs(self.save_dir)

        if "mat" in self.algorithm_name or "ncp" in self.algorithm_name:
            from hetmarl.algorithms.amat_trainer import AMATTrainer as TrainAlgo
            from hetmarl.algorithms.transformer_policy import TransformerPolicy as Policy
        elif "ft" in self.algorithm_name:
            pass
        else:
            raise NotImplementedError

        # share_observation_space = self.envs.share_observation_space[
        #     0] if self.use_centralized_V else self.envs.observation_space[0]
        share_observation_space = self.envs.observation_space[0] # Observation_space and shared_observation_space are the same in our case
        
        print("obs_space: ", self.envs.observation_space)
        # print("share_obs_space: ", self.envs.share_observation_space)
        print("act_space: ", self.envs.action_space)


        if "ft" not in self.algorithm_name:
            # policy network
            if "mat" in self.algorithm_name or "ncp" in self.algorithm_name:
                self.policy = Policy(self.all_args,
                                self.envs.observation_space[0],
                                share_observation_space,
                                self.envs.action_space[0],
                                self.num_agents,
                                self.n_agent_types,
                                device=self.device)
            else:
                self.policy = Policy(self.all_args,
                                self.envs.observation_space[0],
                                share_observation_space,
                                self.envs.action_space[0],
                                device=self.device)
                
            if self.model_dir is not None:
                self.restore(self.model_dir)

            # algorithm
            self.trainer = TrainAlgo(self.all_args, self.policy, self.num_agents, device=self.device)

            # buffer
            self.buffer = SharedReplayBuffer(self.all_args,
                                            self.num_agents,
                                            self.envs.observation_space[0],
                                            share_observation_space,
                                            self.envs.action_space[0],
                                            self.all_args.env_name)

    def run(self):
        """Collect training data, perform training updates, and evaluate policy."""
        raise NotImplementedError

    def warmup(self):
        """Collect warmup pre-training data."""
        raise NotImplementedError

    def collect(self, step):
        """Collect rollouts for training."""
        raise NotImplementedError

    def insert(self, data):
        """
        Insert data into buffer.
        :param data: (Tuple) data to insert into training buffer.
        """
        raise NotImplementedError

    @torch.no_grad()
    def compute(self):
        """Calculate returns for the collected data."""
        self.trainer.prep_rollout()
        if self.buffer.available_actions is None:
            next_values = self.trainer.policy.get_values(np.concatenate(self.buffer.share_obs[-1]),
                                                         np.concatenate(self.buffer.obs[-1]),
                                                         np.concatenate(self.buffer.rnn_states_critic[-1]),
                                                         np.concatenate(self.buffer.masks[-1]))
        else:
            next_values = self.trainer.policy.get_values(np.concatenate(self.buffer.share_obs[-1]),
                                                         np.concatenate(self.buffer.obs[-1]),
                                                         np.concatenate(self.buffer.rnn_states_critic[-1]),
                                                         np.concatenate(self.buffer.masks[-1]),
                                                         np.concatenate(self.buffer.available_actions[-1])) 
        next_values = np.array(np.split(_t2n(next_values), self.n_rollout_threads))
        self.buffer.compute_returns(next_values, self.trainer.value_normalizer)

    def train(self):
        """Train policies with data in buffer. """
        self.trainer.prep_training()
        train_infos = self.trainer.train(self.buffer)      
        self.buffer.after_update()
        return train_infos

    def save(self, episode, save_separately=False):
        """Save policy's actor and critic networks."""
        self.policy.save(self.save_dir, episode)

    def restore(self, model_dir):
        """Restore policy's networks from a saved model."""
        self.policy.restore(model_dir)
 
    def log_train(self, train_infos, total_num_steps):
        """
        Log training info.
        :param train_infos: (dict) information about training update.
        :param total_num_steps: (int) total number of training env steps.
        """
        for k, v in train_infos.items():
            if self.use_wandb:
                wandb.log({k: v}, step=total_num_steps)
            else:
                self.writter.add_scalars(k, {k: v}, total_num_steps)

    def log_env(self, env_infos, total_num_steps):
        """
        Log env info.
        :param env_infos: (dict) information about env state.
        :param total_num_steps: (int) total number of training env steps.
        """
        for k, v in env_infos.items():
            if len(v)>0:
                if self.use_wandb:
                    wandb.log({k: np.mean(v)}, step=total_num_steps)
                else:
                    self.writter.add_scalars(k, {k: np.mean(v)}, total_num_steps)
