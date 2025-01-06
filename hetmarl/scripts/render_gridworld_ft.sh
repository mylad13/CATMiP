#!/bin/sh
env="GridWorld"
scenario="MiniGrid-SearchAndRescue-v0"
num_agents=2
num_obstacles=30
algo="ft_nearest"
exp="Render_ft"
seed_max=1


echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`
do
    CUDA_VISIBLE_DEVICES=0 python render/render_gridworld_ft.py\
      --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --scenario_name ${scenario} \
      --num_agents ${num_agents} --num_obstacles ${num_obstacles} --seed 1 --n_rollout_threads 1 \
      --max_steps 300 --agent_view_size 7 --local_step_num 10 \
      --astar_cost_mode normal --grid_size 15 --wandb_name "baselines" --user_name "mylad"  \
      --use_wandb --action_size 3 --use_eval --n_eval_rollout_threads 1 --eval_episodes 100 \
      --n_agent_types 2 --agent_types_list 0 1 --detect_traces --use_agent_obstacle \
      --use_partial_comm --com_sigma 1 --asynch \
      # --use_render --render_episodes 30  \
      #removed  --save_gifs --block_doors --block_chance 0.6 
done


# NOTES:
# n_rollout_threads should be the same as n_eval_rollout_threads for eval to work properly for now.