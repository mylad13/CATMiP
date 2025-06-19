from gym.envs.registration import register as gym_register

env_list = []

def register(
    id,
    grid_size,
    max_steps,
    local_step_num,
    agent_view_size,
    num_obstacles,
    num_agents,
    n_agent_types,
    agent_types_list,
    agent_pos,
    entry_point,
    reward_threshold=0.95,
    use_full_comm=False,
    use_partial_comm=False,
    use_orientation = False,
    use_same_location = True,
    use_agent_obstacle = False,
    use_time_penalty = False,
    use_energy_penalty = False,
    use_intrinsic_reward = False,
    use_slam_noise = False,
    slam_noise_prob = 0.05,
    algorithm_name = 'amat',
    trajectory_forget_rate = 0.8,
    com_sigma = 5,
    block_doors = False,
    block_chance = 0.25,
    detect_traces = False,
    action_size = 5
):
    assert id.startswith("MiniGrid-")
    # assert id not in env_list

    # Register the environment with OpenAI gym
    gym_register(
        id=id,
        entry_point=entry_point,
        kwargs={
        'grid_size': grid_size,
        'max_steps': max_steps,
        'local_step_num': local_step_num,
        'agent_view_size': agent_view_size,
        'num_obstacles': num_obstacles,
        'num_agents': num_agents,
        'n_agent_types': n_agent_types,
        'agent_types_list' : agent_types_list,
        'agent_pos': agent_pos,
        'use_full_comm': use_full_comm,
        'use_partial_comm': use_partial_comm,
        'use_orientation':use_orientation,
        'use_same_location': use_same_location,
        'use_agent_obstacle': use_agent_obstacle,
        'use_time_penalty': use_time_penalty,
        'use_energy_penalty': use_energy_penalty,
        'use_intrinsic_reward': use_intrinsic_reward,
        'use_slam_noise': use_slam_noise,
        'slam_noise_prob': slam_noise_prob,
        'algorithm_name': algorithm_name,
        'trajectory_forget_rate': trajectory_forget_rate,
        'com_sigma': com_sigma,
        'block_doors': block_doors,
        'block_chance': block_chance,
        'detect_traces': detect_traces,
        'action_size': action_size
        },
        reward_threshold=reward_threshold
    )

    # Add the environment to the set
    env_list.append(id)
