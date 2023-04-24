import torch
from torch.utils.data import Dataset
import json
import numpy as np
import argparse
import os
from robosuite.synthetic_comparisons import generate_synthetic_comparisons_commands

parser = argparse.ArgumentParser(description='')

parser.add_argument('--policy-dir', type=str, default='', help='')
parser.add_argument('--output-dir', type=str, default='', help='')

args = parser.parse_args()

policy_dir = args.policy_dir
output_dir = args.output_dir

print("GETTING TRAJECTORY ROLLOUTS...")
trajectories = []
trajectory_rewards = []
for config in os.listdir(policy_dir):
    policy_path = os.path.join(policy_dir, config)
    if os.path.isdir(policy_path) and os.listdir(policy_path):  # Check that policy_path is a directory and that directory is not empty
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
print("NUM_TRAJECTORIES:", num_trajectories)
print("COMPILING DATASET:")
dataset_traj_as = []
dataset_traj_bs = []
dataset_comps = []
for i in range(0, num_trajectories):
    print("GENERATING COMPARISONS FOR i =", i)
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

        for c in gt_reward_ordinary_comps+speed_ordinary_comps+height_ordinary_comps+distance_to_bottle_ordinary_comps+distance_to_cube_ordinary_comps:
            dataset_traj_as.append(traj_i)
            dataset_traj_bs.append(traj_j)
            dataset_comps.append(c)

        for c in gt_reward_flipped_comps+speed_flipped_comps+height_flipped_comps+distance_to_bottle_flipped_comps+distance_to_cube_flipped_comps:
            dataset_traj_as.append(traj_j)
            dataset_traj_bs.append(traj_i)
            dataset_comps.append(c)

np.save(os.path.join(output_dir, 'traj_as.npy'), dataset_traj_as)
np.save(os.path.join(output_dir, 'traj_bs.npy'), dataset_traj_bs)
with open(os.path.join(output_dir, 'nlcomps.json'), 'w') as f:
    json.dump(dataset_comps, f)



