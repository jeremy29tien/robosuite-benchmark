import torch
from torch.utils.data import Dataset
import json
import numpy as np
import argparse
import os
import torch.nn.functional as F
from nl_traj_feature_learning.learn_features import NLTrajAutoencoder
from nl_traj_feature_learning.nl_traj_dataset import NLTrajComparisonDataset
from gpu_utils import determine_default_torch_device

parser = argparse.ArgumentParser(description='')

parser.add_argument('--reference-policy-dir', type=str, default='', help='')
parser.add_argument('--policy-dir', type=str, default='', help='')
parser.add_argument('--trajs-per-policy', type=int, default=5, help='')

args = parser.parse_args()

reference_policy_dir = args.reference_policy_dir
policy_dir = args.policy_dir
trajs_per_policy = args.trajs_per_policy

# Load model
device = torch.device(determine_default_torch_device(not torch.cuda.is_available()))
print("device:", device)
model = torch.load('/home/jeremy/robosuite-benchmark/models/3/model.pth')
model.to(device)

################################################
### TEMPORARY ANALYSIS OF LANGUAGE EMBEDDINGS ##
################################################
data_dir = '/home/jeremy/robosuite-benchmark/data/nl-traj/62x3all-pairs'
model.eval()
# Some file-handling logic first.
train_nlcomp_file = os.path.join(data_dir, "train/nlcomps.npy")
val_nlcomp_file = os.path.join(data_dir, "val/nlcomps.npy")
train_traj_a_file = os.path.join(data_dir, "train/traj_as.npy")
train_traj_b_file = os.path.join(data_dir, "train/traj_bs.npy")
val_traj_a_file = os.path.join(data_dir, "val/traj_as.npy")
val_traj_b_file = os.path.join(data_dir, "val/traj_bs.npy")

train_dataset = NLTrajComparisonDataset(train_nlcomp_file, train_traj_a_file, train_traj_b_file,
                                        preprocessed_nlcomps=True)
# val_dataset = NLTrajComparisonDataset(val_nlcomp_file, val_traj_a_file, val_traj_b_file,
#                                       preprocessed_nlcomps=True)
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=256, shuffle=True, num_workers=4, pin_memory=True
)
# val_loader = torch.utils.data.DataLoader(
#     val_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True
# )

encoded_traj_diffs = []
encoded_langs = []
dot_prods = []
for train_datapoint in train_loader:
    with torch.no_grad():
        traj_a, traj_b, lang = train_datapoint
        traj_a = torch.as_tensor(traj_a, dtype=torch.float32, device=device)
        traj_b = torch.as_tensor(traj_b, dtype=torch.float32, device=device)
        lang = torch.as_tensor(lang, dtype=torch.float32, device=device)
        train_datapoint = (traj_a, traj_b, lang)
        pred = model(train_datapoint)

        encoded_traj_a, encoded_traj_b, encoded_lang, decoded_traj_a, decoded_traj_b = pred

        encoded_traj_diff = (encoded_traj_b - encoded_traj_a).detach().cpu().numpy()
        encoded_traj_diffs.append(encoded_traj_diff)
        encoded_langs.append(encoded_lang.detach().cpu().numpy())
        # print("encoded_traj_b - encoded_traj_a:", encoded_traj_diff[0])
        # print("encoded_lang:", encoded_lang.detach().cpu().numpy()[0])
        dot_prod = torch.einsum('ij,ij->i', encoded_traj_b - encoded_traj_a, encoded_lang)
        dot_prods.append(dot_prod.detach().cpu().numpy())
        # print("dot_prod:", dot_prod.detach().cpu().numpy()[0])

encoded_traj_diffs = np.asarray(encoded_traj_diffs)
print("shape:", encoded_traj_diffs.shape)
encoded_traj_diffs = np.reshape(encoded_traj_diffs, (encoded_traj_diffs.shape[0]*encoded_traj_diffs.shape[1], encoded_traj_diffs.shape[2]))
encoded_langs = np.asarray(encoded_langs)
encoded_langs = np.reshape(encoded_langs, (encoded_langs.shape[0]*encoded_langs.shape[1], encoded_langs.shape[2]))
dot_prods = np.asarray(dot_prods)
dot_prods = np.reshape(dot_prods, (dot_prods.shape[0]*dot_prods.shape[1], dot_prods.shape[2]))

encoded_traj_diffs_std = np.std(encoded_traj_diffs, axis=0)
encoded_langs_std = np.std(encoded_langs, axis=0)
dot_prods_std = np.std(dot_prods, axis=0)

print("encoded_traj_diffs_std:", encoded_traj_diffs_std)
print("encoded_langs_std:", encoded_langs_std)
print("dot_prods_std:", dot_prods_std)

print("encoded_traj_diffs_std mean:", np.mean(encoded_traj_diffs_std))
print("encoded_langs_std mean:", np.mean(encoded_langs_std))
print("dot_prods_std mean:", np.mean(dot_prods_std))


exit(1)
################################################
################################################
################################################


reference_policy_obs = np.load(os.path.join(reference_policy_dir, "traj_observations.npy"))
reference_policy_act = np.load(os.path.join(reference_policy_dir, "traj_actions.npy"))
reference_policy_trajs = np.concatenate((reference_policy_obs, reference_policy_act), axis=-1)
reference_policy_traj = reference_policy_trajs[0]  # We just take the first rollout as our reference.

comp_str = "Move faster."
bert_embedding = np.load('/home/jeremy/robosuite-benchmark/data/nl-traj/all-pairs/nlcomps.npy')[2]  # "Move faster is the 3rd string in file.

reference_policy_traj = torch.unsqueeze(torch.as_tensor(reference_policy_traj, dtype=torch.float32), 0)
bert_embedding = torch.unsqueeze(torch.as_tensor(bert_embedding, dtype=torch.float32), 0)
encoded_ref_traj, _, encoded_comp_str, _, _ = model((reference_policy_traj, reference_policy_traj, bert_embedding))

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

        trajs = torch.as_tensor(trajs, dtype=torch.float32)
        encoded_traj, _, _, _, _ = model((trajs, trajs, bert_embedding))

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