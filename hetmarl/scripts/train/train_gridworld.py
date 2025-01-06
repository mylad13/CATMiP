#!/usr/bin/env python
import sys
import os
import wandb
import socket
import setproctitle
import numpy as np
from pathlib import Path

import torch

from hetmarl.config import get_config

from hetmarl.envs.gridworld.GridWorld_Env import GridWorldEnv
from hetmarl.envs.env_wrappers import InfoSubprocVecEnv, InfoDummyVecEnv, ChooseInfoSubprocVecEnv, ChooseInfoDummyVecEnv


def make_train_env(all_args):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "GridWorld":
                env = GridWorldEnv(all_args)
            else:
                print("Can not support the " +
                      all_args.env_name + "environment.")
                raise NotImplementedError
            env.seed(all_args.seed + rank * 1000)
            return env
        return init_env
    if all_args.n_rollout_threads == 1:
        return InfoDummyVecEnv([get_env_fn(0)])
    else:
        return InfoSubprocVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])


def make_eval_env(all_args):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "GridWorld":
                env = GridWorldEnv(all_args)
            else:
                print("Can not support the " +
                      all_args.env_name + "environment.")
                raise NotImplementedError
            env.seed(all_args.seed * 50000 + rank * 10000)
            return env
        return init_env
    if all_args.n_eval_rollout_threads == 1:
        return InfoDummyVecEnv([get_env_fn(0)])
    else:
        return InfoSubprocVecEnv([get_env_fn(i) for i in range(all_args.n_eval_rollout_threads)])


def parse_args(args, parser):
    parser.add_argument('--scenario_name', type=str, default='simple_spread', help="Which scenario to run on")
    parser.add_argument('--num_agents', type=int, default=2, help="number of players")
    parser.add_argument('--num_obstacles', type=int, default=1, help="number of obstacles")
    parser.add_argument('--agent_pos', type=list, default = None, help="agent_pos")
    parser.add_argument('--grid_size', type=int, default=19, help="map size")
    parser.add_argument('--agent_view_size', type=int, default=7, help="depth the agent can view")
    parser.add_argument('--max_steps', type=int, default=100, help="maximum steps in each episode")
    parser.add_argument('--local_step_num', type=int, default=3, help="steps between macro actions in synchronous mode")
    parser.add_argument("--use_same_location", action='store_true', default=False,
                        help="place the agents in the same location")   
    parser.add_argument("--use_time_penalty", action='store_true', default=False,
                        help="use time penalty")
    parser.add_argument("--use_energy_penalty", action='store_true', default=False,
                        help="use energy penalty")
    parser.add_argument("--use_intrinsic_reward", action='store_true', default=False,
                        help="use intrinsic reward")
    parser.add_argument("--use_orientation", action='store_true', default=False,
                        help="use agents' orientation info")
    parser.add_argument('--use_agent_obstacle',action='store_true', default=False,
                        help="agents are treated as obstacles during navigation")
    
    # eval by time step
    parser.add_argument('--use_time', default=False, action='store_true')
    parser.add_argument('--max_timestep', default=200., type=float)
    


    all_args = parser.parse_known_args(args)[0]

    return all_args


def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)
    
    if all_args.algorithm_name == "rmappo" or all_args.algorithm_name == "rmappg":
        assert (all_args.use_recurrent_policy or all_args.use_naive_recurrent_policy), ("check recurrent policy!")
    elif all_args.algorithm_name == "mappo" or all_args.algorithm_name == "mappg":
        assert (all_args.use_recurrent_policy == False and all_args.use_naive_recurrent_policy == False), ("check recurrent policy!")
    elif all_args.algorithm_name == "amat" or all_args.algorithm_name == "mat" or all_args.algorithm_name == "mancp":
        pass
    else:
        raise NotImplementedError

    # cuda
    if all_args.cuda and torch.cuda.is_available():
        print("choose to use gpu...")
        device = torch.device("cuda:0")
        torch.set_num_threads(all_args.n_training_threads)
        if all_args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        print("choose to use cpu...")
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)

    # torch.cuda.empty_cache() # This might help with low memory issues

    # run dir
    run_dir = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[
                   0] + "/results") / all_args.env_name / all_args.scenario_name / all_args.algorithm_name / all_args.experiment_name
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    # wandb
    if all_args.use_wandb:
        run = wandb.init(config=all_args,
                         project=all_args.env_name,
                         entity=all_args.wandb_name,
                         notes=socket.gethostname(),
                         name=str(all_args.algorithm_name) + "_" +
                         str(all_args.experiment_name) +
                         "_seed" + str(all_args.seed),
                         group=all_args.scenario_name,
                         dir=str(run_dir),
                         job_type="training",
                         reinit=True)
    else:
        if not run_dir.exists():
            curr_run = 'run1'
        else:
            exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in run_dir.iterdir() if str(folder.name).startswith('run')]
            if len(exst_run_nums) == 0:
                curr_run = 'run1'
            else:
                curr_run = 'run%i' % (max(exst_run_nums) + 1)
        run_dir = run_dir / curr_run
        if not run_dir.exists():
            os.makedirs(str(run_dir))

    setproctitle.setproctitle(str(all_args.algorithm_name) + "-" + \
        str(all_args.env_name) + "-" + str(all_args.experiment_name) + "@" + str(all_args.user_name))

    # seed
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    # env init
    envs = make_train_env(all_args)
    eval_envs = make_eval_env(all_args) if all_args.use_eval else None
    num_agents = all_args.num_agents
    if all_args.asynch:
        # if all_args.use_full_comm:
        all_args.episode_length = all_args.max_steps
        # elif all_args.use_partial_comm:
        #     all_args.episode_length = all_args.max_steps * all_args.num_agents
    else:
        # if all_args.use_full_comm:
        all_args.episode_length = all_args.max_steps // all_args.local_step_num
        # elif all_args.use_partial_comm:
            # all_args.episode_length = all_args.max_steps // all_args.local_step_num * all_args.num_agents

    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "num_agents": num_agents,
        "device": device,
        "run_dir": run_dir
    }

    # run experiments
    from hetmarl.runner.shared.gridworld_runner import GridWorldRunner as Runner

    runner = Runner(config)
    runner.run()
    
    # post process
    envs.close()
    if all_args.use_eval and eval_envs is not envs:
        eval_envs.close()

    if all_args.use_wandb:
        run.finish()
    else:
        runner.writter.export_scalars_to_json(str(runner.log_dir + '/summary.json'))
        runner.writter.close()


if __name__ == "__main__":
    main(sys.argv[1:])
