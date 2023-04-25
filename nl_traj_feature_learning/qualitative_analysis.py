import torch
from torch.utils.data import Dataset
import json
import numpy as np
import argparse
import os
import torch.nn.functional as F
from nl_traj_feature_learning.learn_features import NLTrajAutoencoder

parser = argparse.ArgumentParser(description='')

parser.add_argument('--reference-policy-dir', type=str, default='', help='')
parser.add_argument('--policy-dir', type=str, default='', help='')
parser.add_argument('--trajs-per-policy', type=int, default=5, help='')

args = parser.parse_args()

reference_policy_dir = args.reference_policy_dir
policy_dir = args.policy_dir
trajs_per_policy = args.trajs_per_policy

# Load model
model = torch.load('/home/jeremy/robosuite-benchmark/models/model.pth')

reference_policy_obs = np.load(os.path.join(reference_policy_dir, "traj_observations.npy"))
reference_policy_act = np.load(os.path.join(reference_policy_dir, "traj_actions.npy"))
reference_policy_trajs = np.concatenate((reference_policy_obs, reference_policy_act), axis=-1)
reference_policy_traj = reference_policy_trajs[0]  # We just take the first rollout as our reference.

comp_str = "Move faster."
bert_embedding = np.load('/home/jeremy/robosuite-benchmark/data/nl-traj/all-pairs/nlcomps.npy')[2]  # "Move faster is the 3rd string in file.

encoded_ref_traj, _, encoded_comp_str, _, _ = model((torch.as_tensor(reference_policy_traj, dtype=torch.float32), torch.as_tensor(reference_policy_traj, dtype=torch.float32), torch.as_tensor(bert_embedding, dtype=torch.float32)))

# This is the traj we are looking for.
encoded_target_traj = encoded_ref_traj + encoded_comp_str

print("GETTING TRAJECTORY ROLLOUTS...")

max_similarity = 0
max_sim_policy = ''
for config in os.listdir(policy_dir):
    policy_path = os.path.join(policy_dir, config)
    if os.path.isdir(policy_path) and os.listdir(policy_path):  # Check that policy_path is a directory and that directory is not empty
        # print(policy_path)
        observations = np.load(os.path.join(policy_path, "traj_observations.npy"))
        actions = np.load(os.path.join(policy_path, "traj_actions.npy"))
        rewards = np.load(os.path.join(policy_path, "traj_rewards.npy"))
        # observations has dimensions (n_trajs, n_timesteps, obs_dimension)
        trajs = np.concatenate((observations, actions), axis=-1)

        # Downsample
        trajs = trajs[0:trajs_per_policy]
        rewards = rewards[0:trajs_per_policy]

        encoded_traj, _, _, _, _ = model((torch.as_tensor(trajs, dtype=torch.float32), torch.as_tensor(trajs, dtype=torch.float32), torch.as_tensor(bert_embedding, dtype=torch.float32)))

        for i in range(trajs_per_policy):
            similarity = F.cosine_similarity(encoded_target_traj, encoded_traj[i])
            similarity = similarity.item()
            if similarity > max_similarity:
                max_similarity = similarity
                max_sim_policy = policy_path + ' ' + str(i)
                print("max sim so far at:", policy_path)
                print("i:", i)


print("policy with best sim:", max_sim_policy)
print("max sim:", max_similarity)