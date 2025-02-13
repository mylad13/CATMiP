#!/bin/sh
env="GridWorld"
scenario="MiniGrid-SearchAndRescue-v0"
num_agents=3
num_obstacles=30
algo="amat"
exp="Render_MAT"
seed_max=1


echo "env is ${env}"
for seed in `seq ${seed_max}`
do
    CUDA_VISIBLE_DEVICES=0 python render/render_gridworld.py\
      --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --scenario_name ${scenario} \
      --num_agents ${num_agents} --num_obstacles ${num_obstacles} --seed 2 --n_rollout_threads 1 \
      --max_steps 500 --agent_view_size 7 --local_step_num 10 \
      --astar_cost_mode normal --model_dir "/home/farjadnm/Het-TeamSAR/hetmarl/scripts/results/GridWorld/MiniGrid-SearchAndRescue-v0/amat/MAT_for_target_finding/wandb/run-20250117_140734-1y3l6znp/files" \
      --grid_size 20 --wandb_name "mylad" \
      --user_name "mylad"  \
      --use_wandb  --use_action_masking --action_size 3 \
      --trajectory_forget_rate 0.9115 --goal_history_decay 0 \
      --n_head 1 --n_embd 192 --n_block 1 --recurrent_hidden_size 192 \
      --use_eval --n_eval_rollout_threads 1 --eval_episodes 100 \
      --n_agent_types 2 --agent_types_list 0 1 1 --use_energy_penalty --use_agent_obstacle --detect_traces \
      --use_full_comm --com_sigma 2  --asynch \
      --use_render --render_episodes 30 \
      #removed  --save_gifs --block_doors --block_chance 0.6  --use_time_penalty --use_intrinsic_reward
done

# NOTES:
# n_rollout_threads should be the same as n_eval_rollout_threads for eval to work properly for now.