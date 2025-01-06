#!/bin/sh
env="GridWorld"
scenario="MiniGrid-SearchAndRescue-v0"
num_agents=3
num_obstacles=30
algo="amat"
exp="MAT_for_target_finding"
seed_max=1

# run-20240508_111918-lhsdnu44 is the first one with async and full communication: 96 threads with ppo_epoch 8
# run-20240510_100611-dac9doe8 is async and full communication: 64 threads with ppo_epoch 8
# run-20240513_113438-ye7z9cmb is the first one with sync and partial communication (com_distance = 10): 32 threads with ppo_epoch 4
# run-20240506_102417-583ta3yn is the best one with sync and full communication: 96 threads with ppo_epoch 10

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=0 python train/train_gridworld.py \
    --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --scenario_name ${scenario} \
    --log_interval 1 --wandb_name "mylad" --user_name "mylad" --num_agents ${num_agents} \
    --num_obstacles ${num_obstacles} --seed 2 --n_training_threads 4 \
    --n_rollout_threads 64 --num_mini_batch 1 --num_env_steps 800000000 --ppo_epoch 10 \
    --lr 1e-4 --use_linear_lr_decay  --max_steps 200 --agent_view_size 7 --local_step_num 10 \
    --astar_cost_mode normal --grid_size 20 --use_full_comm  --entropy_coef 0.01 \
    --clip_param 0.05 --gamma 1 --asynch \
    --trajectory_forget_rate 0.9145 --goal_history_decay 0 \
    --n_head 1 --n_embd 192 --n_block 1 --action_size 3 --recurrent_hidden_size 192 \
    --n_agent_types 2 --agent_types_list 0 1 10  --use_action_masking  \
    --use_energy_penalty --use_agent_obstacle --detect_traces \
     #removed  --block_doors --block_chance 0.3 --use_time_penalty  --use_intrinsic_reward 
    #  --model_dir "./results/GridWorld/MiniGrid-SearchAndRescue-v0/amat/MAT_for_target_finding/wandb/run-20241125_140425-iog1awmy/files"
done

# 0.2269: outchannels = in_channels**2, padding = 1
# 0.2270: n_embd was increased to 384 from 192, n_head still being 3
# 0.2271: only give target_not_found penalty when agents are not moving, removed target_found reward, reduced main_agent penalty for not moving towards target from 0.25 (0.5 for going forward in the wrong direction) to 0.05. if this was worse, revert the main_agent penalty back to 0.25
# 0.2272: main agent is now 0.25 penalized for not getting close to the target every time step
# 0.2273: decreased n_head to 2. the result was worse, now going down to 1
# 0.2274: n_head is now 1, n_embd is 384, n_block is 1. the initial result was better than 2 heads, but still worse than 3 heads. now going to try 4 heads
# 0.2275: n_head is now 4, n_embd is 384, n_block is 1.
# 0.2276: n_head is now 4, n_embd is 560, n_block is 1.
# 0.2277: n_head is now 3, n_embd is 384, n_block is 2.
# 0.2278: n_head is now 3, n_embd is 384, n_block is 1, added a new conv2d layer.
# 0.2279: added penalty for running towards obstacles, added obstacles as observations again, removed use_agent_obstacle (this was wrong cuz of pythonpath issues)
# 0.22791: fixed the pythonpath issue, which means changes in 0.2278 and 0.2279 are now in effect simultaneously
# 0.2251: back to using 1 conv2d layer.
# 0.2252: revert obstacle_all_map to 1s instead of 0s
# 0.2253: decreased ppo_epochs to 3
# 0.2254: decreased ppo_epochs to 1
# 0.2255: increased ppo_epochs to 7 (this was better than 1,3,5, but slower)
# 0.2256: agent_id_embedding is done by concatenating one-hot vectors with n_agent size instead of learning an embedding
# 0.2257: one-hot embedding is now agent_type, instead of just agent_id. also added it to the actions, and action encoding no longer using conv2d
## 0.2258: changed --n_head 1 --n_embd 100 from 3 and 384 (GOOD RESULT)
# 0.2259: not cutting the predicted path at local_step_num, and reduced goal history decay to 0.5 from 0.75, increased path prediction decay to 0.8 from 0.6 (BAD)
# 0.2260: increased goal history decay to 0.85 from 0.5, path prediction decay went back to 0.6
# 0.2261: decreased n_training_threads to 1 from 4 (converged a bit later) (5297 Mb memory usage) (FPS 2332)
# 0.2262: increased n_training_threads to 8 from 1 (5330 Mb memory usage) (FPS 2196) (this was worse than 1 and 4, will switch to 2 later)
# 0.2263: added very small energy cost for moving (0.01*penalty) (this is going well, so I'm gonna add another change)
# 0.2264: decrease n_training_threads to 2
# 0.2265: increased kernel size to 4
# 0.2266: reverted kernel size back to 3, decrease path prediction decay to 0 from 0.6, set number of training threads back to 4
# 0.2267: energy penalty is now 0.02 instead of 0.01
# 0.2268: increased n_block to 2
# 0.2269: decreased n_block back to 1, changed the id embedding to learned+sum instead of one-hot+embedding
# 0.2100: energy penalty is now 0.03 instead of 0.02
# 0.2101: energy penalty is now 0.04 instead of 0.03 (this was worse, so I will try to give more reward to exploration)
# 0.2102: exploration reward increased from 0.075 to 0.1 (line 740 of multiexploration)
# 0.2103: exploration reward increased from 0.1 to 0.2 (line 740 of multiexploration) RESULT: 98% accuracy
# 0.2104: removed other agents predicted path (which had 0 decay ratio so it was their location so far)
# 0.2105: removed main agent's pos map from all the inputs
# 0.2106: the two channels are added back. Not giving reward to every agent when the main agent stays at the target now.
# 0.2107: removed the two channels again.
# 0.2108: added the two channels back. adding 0.25 path prediction decay (result is similar to 2106)
# 0.2109: increased path prediction decay to 0.5
# 0.7: trajectory forget rate is now 0.7 instead of 0.2109
# 0.7: path prediction decay is now 0
# 0.701: path prediction decay is now 0.25
# 0.9: trajectory forget rate is 0.9, and replaced predicted paths with other agents trajectories
# 0.903: fixed the observations using target_each_map instead of all_map
# 0.904: fixed the exploration weights after target is found scaling incorrectly. changed it to reward a radius around the target
# 0.905: fixed the bug in the weight function again. removed the x2 scaling.
# 0.906: exploration reward is 0.3 from 0.2 (was worse)
# 0.907: exploration reward is 0.25 from 0.3
# 0.908: max_steps is now 400 from 240
# 0.900: ppo_epochs is now 3 from 7
# 0.909: ppo_epochs is now 5 from 3
# 0.910: ppo_epochs is now 1 from 3
# 0.911: goal history is changed to current goal by setting the decay to 0
# 0.912: goal history decay is now 0.5 (extremely good result)
# 0.913: using partial communication
# 0.914: using partial communication, n_embd is now 150 from 100 (converging much later and struggling to)
# 0.5:  using partial communication, trajectory forget rate is now 0.5 from 0.9
# 0.914: using partial communication, local step num is down to 5 from 10
# 0.915: fixed partial communication to be ACTUALLY partial communication
# 0.916: goal_history_decay is now 0.25 from 0.5
# 0.920: distributed training pogchamp
# 0.9004: distributed with centralized pre-training, agent penalty for staying close is now 0.5 from 0.3. The pre-trained model was trained with 0.3 (didnt converge)
# 0.9005: distributed with centralized pre-training, agent penalty for staying close is now 0.4 from 0.5. The pre-trained model was trained with 0.3 (oscilated, stopped it early to pretrain centrally with 0.4 next)
# 0.9006: training centrally with 0.4 agent penalty for staying close (coverged later but the performance is better)
# 0.9007-9: Removed CNN, and just flattened the observations. Trained centrally using full communication.
# 0.9010: Using ChannelViT as the observation encoder. Hyperparameters are in obsidian (2024-02-13)
# 0.9011: Added back the classifier token, we are using it as the observation's latent representation. Hyperparameters of CViT: patch_size=8, in_chans=input_channel, embed_dim=n_embd, depth=1, num_heads=1, mlp_ratio=4 (run-20240214_110048-3sxisll2)
# 0.9012: Untied the image filters for ChannelViT
# 0.9013: naive patch aggregator run-20240215_163454-ulum7ftl
# 0.9014: Attention-based patch aggregator
# 0.9015: patch size 16, heads to 3, n_embed to 192
# 0.9016: Using 5 observation channels, n_embed 112, n_head ViT 2, patch size 8
# 0.9017: Using 4 observation channels
# 0.9017: changed patch size to 4, removed penalty for staying close.
# 0.9019: added n_embed_vit 48 with one head, and changed n_embd to 96 from 112
# 0.9018: doubled the reward for staying at the target to 10*_reward, and reaching the target from 1* to 5*
# 0.9020: using 3 heads in ViT, n_embd 96, n_embd_vit 48, added back other agents trajectory map to the observations, so using 5 channels now.
# 0.9021: changed to 1 - exploration reward. 
# 0.9022: changed back to just exploration reward, changed the heads in ViT to 1
# 0.9023: using the previous pre-trained model. changed target_map predictions, changed other_agents_trajectory to all_agents_trajectory
# 0.9024: bold move: remove exploration rewards and energy cost for moving reward
# 0.9025: patch size is 8, n_embd_vit is same as n_embd (96), using all_agents_trajectory_maps which are also now thicker, using agent pos instead of each_agent trajectory,
# 0.9700: changed some reward structures, added time penalty.
# 0.9701: changed all agents trajectory to other agents trajectory (hopefully they will learn to not stay close to each other)
# 0.9702: sharing the obtained rewards between all agents
# 0.9703: changed --lr from 5e-4 to 1e-3
# 0.9704: added back exploration rewards
# 0.9705: removed time penalty
# 0.9706: Not sharing rewards anymore
# 0.9707: added 0.1 time penalty
# 0.9708: added agent trajectory channel back to the observations
# 0.9709: added 0.1 penalty for agents staying closer than 3 cells (I should remove this and just add more overlap penalty)
# 0.9710: changed n_embd to 192 + added agent goal history with 0.25 decay to the channels
# 0.9711: increased reach target reward, removed goal history, removed staying close penalty, increased overlap penalty, decreased n_rollout_threads to 128 from 192
# 0.9712: increased n_head to 2, removed other agents trajectory and replace it by their positions, n_embd is now 288, n_embd_vit is 192 (lr was 1e-3)
# 0.9713: --lr 5e-3 --use_linear_lr_decay --num_env_steps 320000000
# 0.9714: --lr 5e-4 --ppo_epoch 10
# 0.9715: --num_env_steps 210000000 --max_steps 300
# 0.9716: --ppo_epoch 5 
# 0.9717: --n_head_vit 2 (converged later and the result wasn't better than 1 head)
# 0.9718: --n_head_vit 1 --n_head 3 (not better)
# 0.700: --n_head 2 --trajectory_forget_rate 0.700 --num_env_steps 150000000
# 0.7001: using a cnn instead of channelViT
# 0.702: using FCL instead of CNN (add layernorm after flatten for the next run)
# 0.703: added layernorm after flatten (it was bad)
# 0.703: ppo_epochs 10 (very late convergence, didn't converge in 150M steps)
# 0.704: ppo_epochs 5, lr 1-e4
# 0.705: back to CNN
# 0.706: num_env_steps 210000000 and not using exploration rewards
# 0.707: modified _reward (1 - 0.9*(self.step_count / self.max_steps))
# 0.708: removed time penalty and modified _reward (1 - 0.75*(self.step_count / self.max_steps))
# 0.709: modified _reward (1), local step num is 5, time penalty is back
# 0.710: lr 5e-4
# 0.711: n_block is now 2
# 0.712: modified _reward 1 - 0.5*(self.step_count / self.max_steps), --ppo_epoch 10, exploration rewards are back, lr 1e-4, main agent not getting exploration rewards
# 0.713: halved time penalty, doubled entropy_coef to 0.02
# 0.714: main_agent getting exploration rewards again, modified _reward (1 - 0.9*(self.step_count / self.max_steps))
# 0.715: now sharing rewards between agents
# 0.716: n_block is now 1
# 0.717: ppo_epochs 7, CNN kernel_size 1, CNN padding = 0, use_single_reward
# 0.717: lowered cnn output channels from in_channels**2 to in_channels//2 (3) (the result was worse)
# 0.718: increased cnn output channels from in_channels//2 (3) to in_channles*2  (still bad)
# 0.719: decreased learning rate to 5e-5, kernel_size 3, out_channel1 = input_channel**2, padding = 1
# 0.720: entropy_coef back to 0.01, lr 1e-4, ppo_epochs 5, local_step_num 10
# 0.720: added a new FCL to the CNN
# 0.721: out_channel1 = 64, remvoved the 2nd layer.
# 0.722: --n_rollout_threads 64 from 128
# 0.723: back to 128 rollout threads, but fixed the agent trajectory map being empty. (this is superior)
# 0.724: n_head is now 3
# 0.725: --n_embd 432 (not an improvement over 2 heads and 288)
# 0.726: --n_head 2 --n_embd 432 (not working well in the early stages)
# 0.7261: --n_head 2 --n_embd 288, changed the shifted_action variable to concatenate the two action dimensions (not better than 0.723)
# 0.7262: added current goal position as a channel to the observations
# 0.7263: changed gamma to 1
# 0.7264: --lr 5e-5 (not good)
# 0.7265: --lr 1e-4, --gamma 0.997, separated other agent pos channels based on agent type 
# 0.7266: ppo_epochs 10
# 0.7267: removed reward for staying at target after done=true (better)
# 0.7268: ppo_epochs 8 and fixed agent_class_identifier not having 1s (sometimes the target is found but main agent takes a bit to choose it as its goal)
# 0.7269: penalty for moving away from the target for the main agent is based on euclidean distance instead of l1 distance now (this was better, so I changed the dist threshhold for getting a reward to 1.5 from 1)
# 0.7270: --n_rollout_threads 96 and changed the dist threshhold of main agent for getting a reward to 1.5 from 1
# 0.7271: added 0.01 energy cost of moving
# 0.7272: exploration_reward_weight = 1 * (1 - self.unexplored_ratio)
# 0.7000: ASYNCH TRAINING
# 0.7001: decreased ppo_epochs to 4 (has converged to about 0.75 in 5M to 20M steps)
# 0.7002: ppo_epoch 8, n_rollout_threads 64 (15G memory used)
# 0.7003: Sync training with partial comm (10G memory used)
# 0.7004: n_training_threads 1 (20G memory used)
# 0.7005: n_training_threads 8 (4 was ideal)
# 0.7006: fixed Sync training with partial comm n_rollout_threads 32 and ppo_epoch 4 since training was very slow otherwise (11G memory used, FPS 333)
# 0.7007: Sync and full comm, adaptive attention, n_rollout_threads 32, ppo_epoch 4: First implementation of the INTERACT action and engineer robot.
# 0.7008: fixed generation of obstacles and rubble, fixed observations of target and rubble, rewards are still messy. continuing the training from 0.7007
# 0.7009: fixed the rewards, training from scratch --block_chance 0.7
# 0.7010: --block_chance 0.6, _reward is just a flat 1 now, time penalty also removed
# 0.7011: --block_chance 0.5, added penalty for taking irrelevant action, using active agents mask on entroies.
# 0.7012: available_actions implemented, just for the interaction action. block_chance 0.3, rollout_threads 64, ppo_epoch 4
# 0.7013: available actions implemented for the obstacles as well. --num_env_steps 240000000. continuing from 0.7012
# 0.7014: cells with l1distance higher than 10 are counted as unavailable. --num_env_steps 240000000. continuing from 0.7013
# 0.7015: Fixed false mask in training. --num_env_steps 210000000 block_chance 0.2 --asynch
# 0.7017: n_head 3, n_embd 384, --entropy_coef 0.01, standby min and max wait is 1-3 now, --n_rollout_threads 48, ppo_epoch 8, added inventroy encoding to the observations, distance not used in available actions
# 0.7018: learning rate 5e-4, continued from 0.7017
# 0.7019: removed the interact action. learning rate 1e-4. --block_chance 0.3
# 0.7020: different speeds set for the agents. continued from 0.7019. ppo_epoch 6
# 0.7021: limited available actions to only rubble related actions for the engineer
# 0.7022: fixed available actions not updating in the buffer. n_rollout_threads 32, ppo_epoch 4
# 0.7024: attempted to fix issues with the buffer that caused some things to not be inserted into the buffer --n_training_threads 16
# 0.7025: sync training
# 0.7026: fixed the issue with async training!!
# 0.7027: enabled masks and active_masks in async training, to get rid of data after environment is done. reward set to 1 - 0.9*(self.step_count / self.max_steps)
# 0.7028: added trackers for rubble and agent rewards.
# 0.7030: Removed agent_type_embedder in the actions decoder, and removed agent trajectory from the observations. asynch training. 64 rollout threads, 4 ppo epochs
# 0.7031: addeded one more agent of type 2, removed standby mode, rollout threads 32, ppo epochs 4, fresh start
# 0.7032: changed the rewards (Removed reaching target reward)
# 0.7034: fixed the problem with the data from the last time agents activate going into asynch training.
# 0.7035: rest time is around 20 for asynch_control, agents activate at information about target being found, modified the rewards 
        # (removed cost of moving, increased cost of carrying rubble, increased cost of running into walls, removed time penalty), gamma = 0.99
# 0.7036: engineer agents activate when their number of known rubbles change
# 0.7037: engineer agents move at the same speed as actuator agents --n_head 2 --n_embd 256 starting from scratch
# 0.7038: n_heads 3 n_embd 384, lr 3e-4, added agent's front pos to observations, removed carrying and inventory stuff
# 0.7039: lr 1e-4, entropy_coef 0.1 (6.1GB memory usage
# 0.7040: added a size 1 kernel layer to the beginning of Obs encoder (9.5GB memory usage?)
# 0.7041: n_embd 192, n_head 3 (14GB memory usage)
# 0.7042: n_embd 192, n_head 3, added RELU after first conv2d and removed the RELU before the fully connected layer, made activate=True in the linear layer of obs_encoder and action_encoder
# 0.7043: fixed how agents become standby and active, --entropy_coef 0.05, made target and rubble maps show 1 in all adjacent cells, added 0.05 movement energy cost
# 0.7044: maps are more asymmetric now. reward is now flat, but increased movement energy cost to 0.5 (tenfold). not restricting engineer actions when number of rubbles < number of engs. continuing from 0.7043
# 0.7045: --max_steps 250, --num_env_steps 200000000, reward for target and rubble is flat 50 now (to make gradient norms smaller), --gamma 0.997
# 0.7046: unreachable goals set action to stop. 
# 0.7047: added traces and 50 reward for finding the target, exploration_reward_weight min is 0.5, stopping penalty is 5
# 0.70482: 1- occupied map is used. removed penalties, _reward is now 1 - 0.75*(self.step_count / self.max_steps).
# 0.70483: Team rewards added, pure exp rewards used (overlap not rewarded).
# 0.7049: n_embd 384
# 0.70491: n_head 2
# 0.70492: n-head 2, now using only one layer of CNN of kernel size 1
# 0.70493: n_block 2
# 0.70494: removed other agent pos maps from observations, --n_head 2 --n_embd 384 --n_block 2
# 0.70495: --lr 5e-5 --num_env_steps 50000000 
# 0.70496: continue from 0.70495, --lr 1e-6, --n_rollout_threads 64, --gamma 1
# 0.70497: --lr 5e-6, n_embd 192, n_head 2, n_block 2, num_agents 3, inversed obstacle map again
# 0.70498: --lr 5e-5, n_embd 192, n_head 2, n_block 1
# 0.70499: time penalty of 1 is added, and rewards are flat now. (time penalty was not added)
# 0.70500: --lr 1e-4, --num_env_steps 25000000  (time penalty was not added)
# 0.70501: time penalty is added
# 0.70502: only team rewards, --lr 5e-4, --num_env_steps 40000000
# 0.70503: kernel size 3, padding 1 
# 0.70504: --lr 5e-3
# 0.70505: --lr 1e-3
# 0.7051: --lr 1e-4, --entropy_coef 0.01 (default), asynch rest time infinity (200), and put agents on standby whenever there is new info (rubbles changed, target found, etc.)
#           looked like it was still improving slowly after 21 million steps, but I stopped it.
# 0.7052: added back second layer of CNN, --n_head 3 --n_embd 288 --num_env_steps 30000000
# 0.7053: --lr 1e-5, --ppo_epoch 10 (did not learn at all)
# 0.7054: --lr 5e-5 (grad norms are still very low, bearly learning, although dist_entropy was decreasing)
# 0.7055: removed the second layer of CNN, added agent class embeddings to the actions, n_embd 96
# 0.7056: --n_embd 96 --n_head 3 --n_block 3 --lr 5e-5 --num_env_steps 30000000
# 0.7057: --lr 1e-4 --num_env_steps 40000000
# 0.7058 + 0.7059: fixed dist_entropy, changed the final MA to end at target_rescued step, (crashed)
# 0.7060: --lr 2e-4
# 0.7061: --lr 1e-4 --num_env_steps 80000000 --ppo_epoch 10
# 0.7062: continuing from 0.7061
# 0.7063: --n_embd 96 --n_head 3 --n_block 1 --lr 5e-5 --num_env_steps 80000000 --ppo_epoch 5, penaling infeasible paths with -5
# 0.7064: fixed how infeasible paths trigger standbys
# 0.7065: ppo_epochs 10
# 0.7066: --n_rollout_threads 128,  ppo_epochs 15, removed other agent location channels (was doing better than previous runs)
# 0.7067: goal_history_decay 1
# 0.7068: goal_history_decay 0, increased trace probability to 0.75 from 0.5, --n_agent_types 2 --agent_types_list 0 1 1
# 0.7069: asynch rest_time = 25, added back other agent pos channels, traces show 0.5 in the target channel, 

# 0.7070: Introducing Map Size Invariance
# 0.7071: --n_head 2 --n_embd 96, --lr 1e-4, not limiting macro action time to 25, limiting it to 50
# 0.7072: asynch rest_time = 25, time penalty removed, _reward is 1-0.9*t , n_rollout_threads 32
# 0.7073: --lr 2e-4, added highlight of action area to obs, --num_env_steps 60000000
# 0.7074: rest_time is 50 again, --lr 5e-4, ppo_epoch 10, --num_env_steps 60000000, _reward is 1 - 0.75t (didnt let this run)
# 0.7075: added 3 hidden layers after the flatten layer at obs encoder
# 0.7076: changed it to 1 hidden layer of size 2000 after the flatten layer at obs encoder, added exploration rewards back, adding distance to target reward for the main agent
# 0.7077: changed target proximity reward structure and n_training_threads to 2 from 4
# 0.7078: continueing from before, training_threads 4 again, min_wait=2, max_wait=4, ending episode if all envs are done, --num_env_steps 40000000
# 0.7079: --num_env_steps 50000000, better distance-to-target reward for agent 0, --lr 1e-3
# 0.7080: --lr 5e-4 (this is better than 1e-3, but still not updating the reward after some point)
# 0.7081: --lr 2e-4 (dist_entropy is decreasing faster, but the reward and target_rescued are doing almost the same as higher learning rates)
# 0.7082: --lr 5e-4, not using action masking
# 0.7083: --lr 2e-4, entropy_coef 0.05, using action masking again (was exactly the same as 0.7081 in the very early stage)
# 0.7084: --lr 5e-4, n_embd 192, n_head 2, --entropy_coef 0.01, --num_env_steps 20000000
# 0.7085: Discrete Action Space with action_size 5, --lr 5e-4, --n_rollout_threads 32 (batch size starting around 5k)
# 0.7086: distance reward is based on astar path length (Initial results not looking good, let's leave this aside for now.) 
# 0.7087: --lr 5e-4, --num_env_steps 30000000, --max_steps 200, l1dist_to_target used for distance reward, --n_rollout_threads 64
# 0.7088: --n_rollout_threads 32
# 0.7089: clipping the max r_exp to 20, added alpha to share r_rescue 80% for the medic and 20% for the explore, r_distance is given in forward and only negatively
# 0.7090: ppo_epoch 15 (dist entropy started increasing after 1.5M steps)
# 0.9001: ppo_epoch 10, adding agent trajectory, hidden_size1 = 9000 instead of 2000
# 0.9002: --lr 3e-4
# 0.9003: using layernorm after conv2d, removed the hidden layer in MLP in the obs_encoder, synch training
# 0.9004:  asynch
# 0.9005:  using adaptive average pooling instead of adaptive max pooling (Also order of agents passing to get_short_term goal was changed but this shouldnt affect anything)
# 0.9006: put the layernorm after the adaptive Max pooling layer, --n_block 2
# 0.9007: --n_block 1
# 0.9008: relu changed to gelu in the obs encoder, --num_env_steps 50000000
# 0.9009: using relu again
# 0.9010: out_channel1 = 32 (performing worse in the early stages)
# 0.9011: out_channel1 is kept at 32, added a hidden layer of size 2000 in the MLP with layernorm after the hidden layer. (GOOD)
# 0.9012: agent activation changed to encourage more shared usage of the transformer
# 0.9013: --max_steps 250, 5 penalty for macro action ending due to stop or infeasible action, trajectory map is only one pixel wide, 
# 0.9014: --n_rollout_threads 64, adding distance reward for main agent (not just distance penalty), target_rescued reward is 200 from 250 
# 0.9015: --lr 5e-4 (worse)
# 0.9016: --lr 1e-4 (bad)
# 0.9017: --lr 3e-4, min_wait=2, max_wait=4, out_channel1=64
# 0.9018: edited the rewards: locate = 50r, rescue = 100r (alpha=0.5), explore = 50/wall_size * n_cells, stop/infeasible = -5
# 0.9019: just made the execution faster by putting computation of agent groups for only the partial comm case (this still plateaus after 5M steps)
# 0.9020: num_mini_batch 4 (high policy loss, low critic grad norm, dist entropy is very slow to decrease)
# 0.9021: ppo_epoch 15
# 0.9022: min_wait=2, max_wait=4, --n_embd 96 --n_block 2, --ppo_epoch 10, out_channel1 = 48, stop/infeasible = -1, rescue = 150r (alpha=0.5)
# 0.9023: --agent_types_list 0 1 1 0 num_agents=4 num_obstacles=5 , 80M steps (very good results, but still not optimal after it finished)
# 0.9024: rest_time=30, new map generation scheme, more reward is given to the agent that finds the target, 5 agents, 2 medics and 3 explorers, grid_size 32, stay reward is 150*0.5, locate reward is 50
# 0.9025: local_map is added to the observations, --n_rollout_threads 128, --num_env_steps 100000000, hidden_size is down to 512 from 2000
# 0.9026: removed the reward part of distance reward, traces are 0.1 in the target map
# 0.9027: fixed_size is 8 from 24, traces are 0.25 in the target map
# 0.9028: fixed_size is 12, stopping penalty is 2
# 0.9029: stopping penalty is 3, alpha=0.75 from 0.5, rescue = 250.
# 0.9030: training with two agents at grid_size 16, 150 max step (working ok!)
# 0.9031: back to previous config, but removed trajecotry and pos maps from local obs
# 0.9032: rescued reward is 400*0.5 (alpha=0.5) (worse)
# 0.9033: rescued reward is 300*0.5 (alpha=0.5) (better but still bad)
# 0.9034: rescued reward is 400*0.75 (alpha=0.75) 
# 0.9035: local_hidden_size1 = global_hidden_size1 = n_embd = 96
# 0.9036: goal history is in local obs instead of global obs (THIS IS DOING GREAT, .9037 is GREAT TOO)
# 0.9037: rescue_reward = 300*alpha for everyone, and 300 for the one who finds the target. same thing with locate reward (50), alpha = 1/self.num_agents
# 0.9038: team rewards are only team rewards (locate = 100/n, rescue = 300/n, explore cap = 50) (converging slowly)
# 0.9039: rescue = 600/n (still bad, the problem is credit assignment)
# just found out that MA termination penalty was always actually 1 not 3
# 0.9040: reward structure back to 9037 (but locate is 100, rescue is 400), but not penalizing macro action termination, only penalizing infeasible actions (3) (end result is slightly worse than 0.9037)
# 0.9041: channel_sizes are 32 instead of 48, local and global obs are concatenated and fed to the newly made obs encoder
# 0.9042: target map only highlights the target cell, not the adjacent cells (converging a bit later, but working just as well)
# 0.9043: other agent positions are removed from the observations (Very similar to 0.9042 up to 3M steps)
# 0.9045: --action_size 4
# 0.9044: --action_size 5 --n_head 1 --n_embd 64 --n_block 1
# 0.9045: layernorms are applied after activation function. CNNs are using ReLU, obs_encoder is using GeLU. obs_encoder starts with layernorm
# 0.9046: layernorms are applied before activation function. removed the hidden layer in obs_encoder
# 0.9047: brought back the hidden layer in obs_encoder
# 0.9048: removed the layernorm at the beginning of the obs_encoder (difference from 0.9044 is the layernorm before final GELU os obs_encoder)
# 0.9049: layernorms are after the GELU at obs_encoder, like the rest of the original MAT. Layernorms in the CNNs are replaced by batchnorm2d (batch sizes are not the same during inference and training so it's a bad idea)
# 0.9050: back to layernorm instead of batchnorm. global encoder has an extra 3x3 convolution layer (16 outs), with layernorm and activation afterwards.
# 0.9051: local encoder has a second cnn layer now (16 outs).
# 0.9052: --lr 5e-4, num_env_steps 80M, distance penalty given at every timestep, not just when going forward. (729306 params)(doing much better) 
# 0.9053: adding one more layer to local and global obs encoders. out channels are now 16-32-16 for both. (611370 params)
# 0.9054: --n_embd 128 (1282282 params)
# 0.9055: --n_head 2 (1282282 params)
# 0.9056: removed individual reward for r_locate
# 0.9057: --n_head 1, n_block 2, n_embd 128 (1547754 params)
# 0.9058: --n_block 1, team reward is r_locate instead of 1/n r_locate (1282282 params)
# 0.9059: Remove one layer of CNN (16-32) (1779626 params, bad)
# 0.9060: 2 layer CNN, 32-16 (1547674 params)
# 0.9061: back to 3 layer CNN (16-32-16), goal history is in both local and global obs (1282570 params)
# 0.9062: 32-64-32 CNN --n_rollout_threads 81 from 128 (2253338 params)
# 0.9063: --lr 8e-4 --n_rollout_threads 96
# 0.9064: 32-64-16 CNN (1683674 params) (doing bad)
# 0.9065: 32-16 CNN, --n_rollout_threads 128 (1548250 params)
# 0.9066: infeasible penalty is 10 from 3
# 0.9067: infeasible penalty is 5, --n_head 3 --n_embd 144
# 0.9068: r_rescue = 300 + 300 for i, r_locate = 50 + 50 for i, _reward is 1-0.9t/h, --lr 4e-4 (1777162 params)
# 0.9069: continue from 9068 because system was rebooted (crashed again)
# 0.9070: --lr 3e-4 
# 0.9071: --action_size 3 (1583554 params) (seems better)
# 0.9072:  min_wait=3, max_wait=5, activation of other standbys is restricted to a randomized wait time. agent pos and trajectory on the global map is 3x3, goal history removed from local map (1583266 params)
# 0.9073: --n_head 1 --n_embd 48 (480130 params)
# 0.9074: --n_head 1 --n_embd 64 , agent_pos_map is single cell again (this was not true until 0.9085), rescue reward is 200*_reward (639666 params)
# 0.9075: --lr 1e-4, fixed_width=fix_height=8, CNN channels are 64-32 and 16-8, macro-action termination penalty is back = 5, n_agents 2 1M1E,
#         --max_steps 150, grid_size 16, --n_rollout_threads 1287, corrected and put adaptivemaxpool2d after the first layer (547338 params, 4.5k initial batch_size, 1000 fps)

# 0.9076: Removed macro action termination penalty, ppo_epoch 15
# 0.9077: --value_loss_coef 0.5 instead of 1
# 0.9078: ppo_epoch 10
# 0.9079: --value_loss_coef 0.1, --num_env_steps 120000000
# 0.9080: changed r_rescue to 150 from 200 (to make the time penalty more effective)
# 0.9081: --num_env_steps 250000000, n_embd 128, value_loss_coef 1 (defualt) (1208522 params)
# 0.9082: n_embd 192  (2025354 params)
# 0.9083: CNNs are 32-16 32-16 to make the flattened local and globals almost the same length (1024 and 784) (1493554 params)
# 0.9084: removed infeasible action penalty (doing okay)
# 0.9085: 20x20 1M2E (1493554 params, 11.8k initial batch_size, 600 fps) (also fixed the agent_pos_map to be 1x1)
# 0.9086: using energy penalty of 0.1 for moving forwards, max_steps 150 (1493554 params, 9k initial batch_size, 620 fps)
# 0.9087: max_steps 200, fixed_size is 4, CNNs are 64, 64-32 (1461714 params, 11.8k initial batch_size, 620 fps)
# 0.9088: n_head 2 (1461714 params) (might have messed up the agent types)
# 0.9089: n_head 1, agents 0 1 10 (1M2E-2M1E) (1493554 params, 11k initial batch_size, 620 fps) (might have messed up the agent types)
# 0.9090: removed the energy penalty for now, will make it less impactful later (1493554 params, 11k initial batch_size, 620 fps) (corrected the agent types)
# 0.9091: n_blocks 4 from 1 (3246162 params, 10.8k initial batch_size, 620 fps)
# 0.9092: n_blocks 1, 1 layer CNNs 16-16 (981810 params, 11k initial batch_size, 636 fps)
# 0.9093: --lr 5e-5, --num_env_steps 150000000 (981810 params, 11k initial batch_size, 636 fps)
# 0.9094: --lr 1e-4, 1 layer CNNs 32-32, no extra individual rewards for r_locate and r_success (1234930 params, 11.2k initial batch_size, 640 fps) (not doing bad, but not great either)
# 0.9095: added separate channels for other agent pos maps, added goal history to local obs, --num_env_steps 200000000 (1236370 params, 11k initial batch_size, 620 fps)
# 0.9096: r_distance is -0.1 instead of -0.5 (worse, reverting to 0.5)
# 0.9097: ppo_epoch 5 instead of 10 (1236370 params, 11k initial batch_size, 680 fps)
# 0.9098: ppo_epoch 15 (1236370 params, 11k initial batch_size, 540 fps)
# 0.9099: ppo_epoch 10, 2 layer CNN 32-16 / 32-16 (1048018 params, 11k initial batch_size, 610 fps)
# 0.9100: CNNs are 64, 64-32, rewards are 100 locate, 300 acquire (1464594 params (1568-1024 local global), 10.7k initial batch_size, 615 fps)

# 0.9101: 20x20, 3 agents. distance reward is -1 instead of -0.5 (1464594 params 1568-1024 local global, 10.7k initial batch_size, 600 fps)
# 0.9102: 16x16, 2 agents. (1464594 params 1568-1024 local global, 6k initial steps, 900 fps)

#0.9107: 20x20, 3 agents, energy penalty of 0.01 is added, max_steps 400, --num_env_steps 500000000
#0.9108: following 0.9107 that was trained for 163M steps, but increasing energy penalty to 0.1 (seems to be learning well, but explorers block pathways when stopping)
#0.9109: following 0.9108 trained for 67M steps, --use_agent_obstacle added to the training
#0.9110: following 0.9109 trained for 482M steps
#0.9111: following 0.9109 trained for 482M steps, lr 5e-5
#0.9112: following 0.9111 trained for 22M steps, lr 25e-6, seed 4
#0.9113: lr 5e-5, seed 3
#0.9114: lr 5e-4, L=4, local_flattened_size2 is 2592, global_flattened_size1 is 1024, Total params is 1679666.

#0.9115: MANCP algorithm (hidden size 64, using CfCCell), same conditions as Model 1,
#        but using energy penalty and agent obstacles, max_steps 300, num_env_steps 500M (1492946 params, 9.4k initial batch_size, 1100 fps)
#0.9116: Added CfCCell with same configs to the decoder as well (1515154 params, 9.4k initial batch_size, 1100 fps)
#0.9117: --recurrent_hidden_size is 192 (1686418 params, 9.1k initial batch_size, 1100 fps)
#0.9118: same settings but with amat instead of mancp (1464594 params, 9.4k initial batch_size, 1100 fps) (the performance of all three (0.9115,0.9917,0.9118) seem to be almost identical)

#0.9119: added engineer agents and rubble to make the env dynamic, grid_size 16, agent_types 0 1 2 (1466130 params, 17.2k initial batch_size, 750 fps)
#0.9120: added rubble_map to the local obs (1466706 params, 17k initial batch_size, 750 fps)
#0.9121: resolution of agent pos, target pos, rubble pos increased in global map, removed goal history from local obs, fixed agent_type2 map (1466130 params, 17.1k initial batch_size, 730 fps)
#0.9122: mancp instead of amat (1687954 params, 23k initial batch_size, 700 fps)
#0.9123: --lr 2e-4 (the progress looks the same, but 1e-4 seems more stable)
#0.9124: lr 1e-4, CNNs are 3-3 layers with 64-128-64 channels (2379826 params, 3136local-1024global, 22.8k initial batch_size, 680 fps)
#0.9125: other agent pos maps are added to the local map (2381554 params, 3136local-1024global, 22.2k initial batch_size, 670 fps)
#0.9126: the global CNN has to have 1 layer to make the map scalable, so back to 1-2 layers with 64 & 64-32 channels (1689682 params, 1568local-1024global, 22.7k initial batch_size, 670 fps)
#0.9127: recurrent_hidden_size and n_embd are both 384 from 192 (4587922 params, 1568local-1024global, 22.7k initial batch_size, 600 fps) (initially better, but it is later worse than 192 n_embd)
#0.9128: n_head 2 (same params and batch_size and fps as 0.9127) (better than 384 and 1 head, but eventually the same as 192 and 1 head)
#0.9129: using amat with n_embd 192 (1467858 params, 1568local-1024global, 22k initial batch_size, 700 fps)
#0.9130: added goal history to local obs (1468434 params, 1568local-1024global, 22.4k initial batch_size, 680 fps)
#0.9131: mancp (1690258 params, 1568local-1024global, 22.6k initial batch_size, 680 fps) (This is no different than amat)
#0.9132: time penalty of 0.2 added.
#0.9133: amat, 2 types of agents, 1 0 10, no time penalty, 128 threads, grid_size 16, max_steps 400 (1468050 params, 21.2k initial batch size, 740 fps after 60 episodes)
#0.9134: 64 threads, max_steps 300, removed rubble channels. (1464594 params, 8.6k initial steps, 700 fps)
#0.9135: removed rescuer agent penalty (same)
#0.9136: halved energy cost for moving to 0.05, made cells surrounded by obstacles unavailable, made cells occupied with agents unavailable. Grid size 18, continuing from previous run.
        # min_wait=2 max_wait=4, rest_time=25
#0.9137: n_embd 384, detect traces are off (4288338 params, 9.2k initial batch size, 620 fps)
#0.91371: made the edges of the map unavailable actions (4288338 params, 8.9k initial batch size, 640 fps)
#0.9138: n_embd 192, global kernel size is 7, global out channels 32 (1568-512, 1280786 params, 9.1k initial batch size, 650 fps) 
#0.91381: 5 ppo_epochs (1568-512, 1280786 params, 9.1k initial batch size, 755 fps)
#0.9139: min_wait=2, max_wait=5, rest_time = 10 (1568-512, 1280786 params, 9.4k initial batch size, 750 fps) (global padding is 1, but it makes more sense for it to be 3)
#0.91391: using ADOPT instead of Adam (upgraded torch to 2.5.1 from 2.4.1) (1568-512, 1280786 params, 9.4k initial batch size, 750 fps) (didn't show any improvement in first 3.5M steps, so will swithc back to Adam)
#0.9140: global_padding_size1 is 3 instead of 1 to compensate for kernel size 7. max_steps is 200, using Adam again (1568-512, 1280786 params, 6.5k initial batch size, 740 fps)
#0.9141: ppo_epochs 10, --num_env_steps 800000000  (1568-512, 1280786 params, 6.5k initial batch size, 620 fps)
#0.9142: traces are generated around the target, grid_size 20x20 (1568-512, 1280786 params, 6.5k initial batch size, 570 fps) 
#0.91431: synchronous training with local_step_num 10 (1568-512, 1280786 params, fixed batch size 3840, 1173 fps)
#0.9144: removed intrinsic rewards from asynch training (1568-512, 1280786 params, 6.5k initial batch size, 570 fps) 
#0.9145: 8 global zones (1568-2048, 1873682 params)



#TODO: change global_kernel_size to 7 from 3
#TODO: add another layer to the obs_encoder

#TODO: try WiredCfCCell 
#TODO: add previous goal to the local map, maybe that helps with collision avoidance?
#TODO: possibly replace separate maps in the global obs with a single map for all other 
#TODO: change the metric to reward/n_agents

#TODO: add other_agent_pos_map to global obs
# change the 2 layers to 64-32 instead of 32-16
# change r_rescue to 100 from 150 (or 300 from 150 and r_locate to 100 from 50)

#TODO: change the learning rate and max number of steps
#TODO: Apply LayerNorm after each convolutional layer or after a couple of convolutional layers, depending on where you observe the most benefit during experimentation.

#TODO: add more depth to the CNNs (for the local map, maybe a 1x1 kernel at the end that reduces the channel size)
#TODO: change GeLUs to ReLUs in the obs encoder
#TODO: possible solution to rewards not increasing after a while with dist_entropy decreasing: increase entropy_coef
#TODO: experiment with removing the agent class encoder from the actions.
#TODO: changed the fixed map size to a lower/higher number

# connection to other agents as part of the observation? Sth is needed to simulate communication loss in training

# increasing the max number of steps will help with learning!
# change the way agent_class_identifier are processed