#!/bin/sh
env="GridWorld"
scenario="MiniGrid-SearchAndRescue-v0"
num_agents=3
num_obstacles=30
algo="amat"
exp="CATMiP_SAR"
seed_max=1


echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=0 python train/train_gridworld.py \
    --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --scenario_name ${scenario} \
    --log_interval 1 --wandb_name "mylad" --user_name "mylad" --num_agents ${num_agents} \
    --num_obstacles ${num_obstacles} --seed 3 --n_training_threads 4 \
    --n_rollout_threads 128 --num_mini_batch 1 --num_env_steps 250000000 --ppo_epoch 10 \
    --lr 1e-4 --use_linear_lr_decay --max_steps 200 --agent_view_size 7 --local_step_num 10 \
    --astar_cost_mode normal --grid_size 20 --use_full_comm  --entropy_coef 0.01 \
    --clip_param 0.05 --gamma 1 --asynch --use_wandb \
    --trajectory_forget_rate 0.9101 --goal_history_decay 0 \
    --n_head 1 --n_embd 192 --n_block 1 --action_size 3 \
    --n_agent_types 2 --agent_types_list 0 1 10 --detect_traces --use_intrinsic_reward --use_action_masking --use_agent_obstacle  \
     #removed --use_time_penalty --use_energy_penalty
    # 
done
