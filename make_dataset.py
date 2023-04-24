import torch
from torch.utils.data import Dataset
import json
import numpy as np
import argparse
import os
from robosuite.synthetic_comparisons import generate_synthetic_comparisons_commands

parser = argparse.ArgumentParser(description='')

parser.add_argument('--policy-dir', type=str, default='', help='')

args = parser.parse_args()

policy_dir = args.policy_dir

trajectories = []
trajectory_rewards = []
for config in os.listdir(policy_dir):
    policy_path = os.path.join(policy_dir, config)
    if os.path.isdir(policy_path):
        observations = np.load(os.path.join(policy_path, "traj_observations.npy"))
        actions = np.load(os.path.join(policy_path, "traj_actions.npy"))
        rewards = np.load(os.path.join(policy_path, "traj_rewards.npy"))
        # observations has dimensions (n_trajs, n_timesteps, obs_dimension)
        trajs = np.concatenate((observations, actions), axis=-1)

        # NOTE: We use extend rather than append because we don't want to add an
        # additional dimension across the policies.
        trajectories.extend(trajs)
        trajectory_rewards.extend(rewards)

trajectories = np.asarray(trajectories)
trajectory_rewards = np.asarray(trajectory_rewards)
num_trajectories = trajectories.shape[0]

for i in range(0, num_trajectories):
    for j in range(i+1, num_trajectories):
        traj_i = trajectories[i]
        traj_j = trajectories[j]
        traj_i_rewards = trajectory_rewards[i]
        traj_j_rewards = trajectory_rewards[j]

        gt_reward_ordinary_comps = generate_synthetic_comparisons_commands(traj_i, traj_j, traj_i_rewards, traj_j_rewards, 'gt_reward')
        gt_reward_flipped_comps = generate_synthetic_comparisons_commands(traj_j, traj_i, traj_j_rewards, traj_i_rewards, 'gt_reward')

        speed_ordinary_comps = generate_synthetic_comparisons_commands(traj_i, traj_j, traj_i_rewards, traj_j_rewards, 'speed')
        speed_flipped_comps = generate_synthetic_comparisons_commands(traj_j, traj_i, traj_j_rewards, traj_i_rewards, 'speed')

        height_ordinary_comps = generate_synthetic_comparisons_commands(traj_i, traj_j, traj_i_rewards, traj_j_rewards, 'height')
        height_flipped_comps = generate_synthetic_comparisons_commands(traj_j, traj_i, traj_j_rewards, traj_i_rewards, 'height')

        distance_to_bottle_ordinary_comps = generate_synthetic_comparisons_commands(traj_i, traj_j, traj_i_rewards, traj_j_rewards, 'distance_to_bottle')
        distance_to_bottle_flipped_comps = generate_synthetic_comparisons_commands(traj_j, traj_i, traj_j_rewards, traj_i_rewards, 'distance_to_bottle')

        distance_to_cube_ordinary_comps = generate_synthetic_comparisons_commands(traj_i, traj_j, traj_i_rewards, traj_j_rewards, 'distance_to_cube')
        distance_to_cube_flipped_comps = generate_synthetic_comparisons_commands(traj_j, traj_i, traj_j_rewards, traj_i_rewards, 'distance_to_cube')



