#!/bin/sh
env="GridWorld"
scenario="MiniGrid-SearchAndRescue-v0"
num_agents=8
num_obstacles=30
algo="amat"
exp="Render_MAT"
seed_max=1

# run-20240906_131232-h7t7t9yp --n_head 1 --n_embd 192 --n_block 1 agent types 0 1 10 and trained for 190M/250M steps
# 0.9101: run-20240918_103712-own2l25c 0 1 10 in 20x20 maps, lr 1e-4, 250M steps finished training.
# 0.9102: run-20240922_112117-hsrxyfdh 0 1 in 16x16 maps, lr 1e-4, 250M steps finished training. 
# Task 3: grid_size 32, max_steps 400, 8 agents, 0 0 1 1 1 1 10 10, seed 36, 2 threads for 500 episodes
echo "env is ${env}"
for seed in `seq ${seed_max}`
do
    CUDA_VISIBLE_DEVICES=0 python render/render_gridworld.py\
      --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --scenario_name ${scenario} \
      --num_agents ${num_agents} --num_obstacles ${num_obstacles} --seed 36 --n_rollout_threads 2 \
      --max_steps 400 --agent_view_size 7 --local_step_num 10 \
      --astar_cost_mode normal --model_dir "./results/GridWorld/MiniGrid-SearchAndRescue-v0/mat/MAT_for_target_finding/wandb/run-20240918_103712-own2l25c/files" \
      --grid_size 32 --wandb_name "mylad" \
      --user_name "mylad" --asynch \
      --use_wandb  --use_action_masking --action_size 3 \
      --trajectory_forget_rate 0.9103 --goal_history_decay 0 \
      --n_head 1 --n_embd 192 --n_block 1 \
      --use_eval --n_eval_rollout_threads 2 --eval_episodes 250 \
      --n_agent_types 2 --agent_types_list 0 0 1 1 1 1 10 10 --detect_traces --use_intrinsic_reward \
      --use_full_comm --com_sigma 2 \
      # --use_render --render_episodes 50 \
      #removed --block_doors --block_chance 0.3 --use_time_penalty  --use_agent_obstacle --use_energy_penalty 
done

# --model_dir "./results/GridWorld/MiniGrid-MultiExploration-v0/mat/MAT_for_target_finding/wandb/run-20240603_160410-pbyq6vvj/files" 
# NOTES:
# n_rollout_threads should be the same as n_eval_rollout_threads for eval to work properly for now.