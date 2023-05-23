import torch
from torch.utils.data import Dataset
import json
import numpy as np
import argparse
import os
import torch.nn as nn
import torch.nn.functional as F
from nl_traj_feature_learning.learn_features import NLTrajAutoencoder
from nl_traj_feature_learning.nl_traj_dataset import NLTrajComparisonDataset
from gpu_utils import determine_default_torch_device
import robosuite.synthetic_comparisons
from robosuite.environments.manipulation.lift_features import speed, height, distance_to_bottle, distance_to_cube
from bert_preprocessing import preprocess_strings


def load_model(model_path):
    # Load model
    device = torch.device(determine_default_torch_device(not torch.cuda.is_available()))
    print("device:", device)
    model = torch.load(model_path)
    model.to(device)
    model.eval()
    return model, device


def print_embedding_statistics(model, device, data_dir, val=False):
    # Some file-handling logic first.
    train_nlcomp_file = os.path.join(data_dir, "train/nlcomps.npy")
    val_nlcomp_file = os.path.join(data_dir, "val/nlcomps.npy")
    train_traj_a_file = os.path.join(data_dir, "train/traj_as.npy")
    train_traj_b_file = os.path.join(data_dir, "train/traj_bs.npy")
    val_traj_a_file = os.path.join(data_dir, "val/traj_as.npy")
    val_traj_b_file = os.path.join(data_dir, "val/traj_bs.npy")

    train_dataset = NLTrajComparisonDataset(train_nlcomp_file, train_traj_a_file, train_traj_b_file,
                                            preprocessed_nlcomps=True)
    val_dataset = NLTrajComparisonDataset(val_nlcomp_file, val_traj_a_file, val_traj_b_file,
                                          preprocessed_nlcomps=True)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=256, shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True
    )

    if val:
        loader = val_loader
    else:
        loader = train_loader

    encoded_traj_diffs = []
    encoded_langs = []
    dot_prods = []
    for datapoint in loader:
        with torch.no_grad():
            traj_a, traj_b, lang = datapoint
            traj_a = torch.as_tensor(traj_a, dtype=torch.float32, device=device)
            traj_b = torch.as_tensor(traj_b, dtype=torch.float32, device=device)
            lang = torch.as_tensor(lang, dtype=torch.float32, device=device)
            datapoint = (traj_a, traj_b, lang)
            pred = model(datapoint)

            encoded_traj_a, encoded_traj_b, encoded_lang, decoded_traj_a, decoded_traj_b = pred

            encoded_traj_diff = (encoded_traj_b - encoded_traj_a).detach().cpu().numpy()
            # print("encoded_traj_diff shape:", encoded_traj_diff.shape)
            if encoded_traj_diffs == []:
                encoded_traj_diffs.append(encoded_traj_diff)
                encoded_traj_diffs = np.squeeze(np.asarray(encoded_traj_diffs))
            else:
                encoded_traj_diffs = np.concatenate((encoded_traj_diffs, encoded_traj_diff), axis=0)

            encoded_lang_np = encoded_lang.detach().cpu().numpy()
            # print("encoded_lang_np shape:", encoded_lang_np.shape)
            if encoded_langs == []:
                encoded_langs.append(encoded_lang_np)
                encoded_langs = np.squeeze(np.asarray(encoded_langs))
            else:
                encoded_langs = np.concatenate((encoded_langs, encoded_lang_np), axis=0)

            # print("encoded_traj_b - encoded_traj_a:", encoded_traj_diff[0])
            # print("encoded_lang:", encoded_lang_np[0])
            dot_prod = torch.einsum('ij,ij->i', encoded_traj_b - encoded_traj_a, encoded_lang)
            dot_prod_np = dot_prod.detach().cpu().numpy()
            if dot_prods == []:
                dot_prods.append(dot_prod_np)
                dot_prods = np.squeeze(np.asarray(dot_prods))
            else:
                dot_prods = np.concatenate((dot_prods, dot_prod_np), axis=0)

            # print("dot_prod:", dot_prod_np[0])

    # encoded_traj_diffs = np.asarray(encoded_traj_diffs)
    print("encoded_traj_diffs shape:", encoded_traj_diffs.shape)
    # encoded_traj_diffs = np.reshape(encoded_traj_diffs, (encoded_traj_diffs.shape[0]*encoded_traj_diffs.shape[1], encoded_traj_diffs.shape[2]))
    # encoded_langs = np.asarray(encoded_langs)
    print("encoded_langs shape:", encoded_langs.shape)
    # encoded_langs = np.reshape(encoded_langs, (encoded_langs.shape[0]*encoded_langs.shape[1], encoded_langs.shape[2]))
    # dot_prods = np.asarray(dot_prods)
    # dot_prods = np.reshape(dot_prods, (dot_prods.shape[0]*dot_prods.shape[1], dot_prods.shape[2]))

    pos_dot_prods = np.asarray([dp for dp in dot_prods if dp > 0])
    neg_dot_prods = np.asarray([dp for dp in dot_prods if dp < 0])

    encoded_traj_diffs_mean = np.mean(encoded_traj_diffs, axis=0)
    encoded_langs_mean = np.mean(encoded_langs, axis=0)
    dot_prods_mean = np.mean(dot_prods, axis=0)
    pos_dot_prods_mean = np.mean(pos_dot_prods, axis=0)
    neg_dot_prods_mean = np.mean(neg_dot_prods, axis=0)

    encoded_traj_diffs_std = np.std(encoded_traj_diffs, axis=0)
    encoded_langs_std = np.std(encoded_langs, axis=0)
    dot_prods_std = np.std(dot_prods, axis=0)
    pos_dot_prods_std = np.std(pos_dot_prods, axis=0)
    neg_dot_prods_std = np.std(neg_dot_prods, axis=0)

    print("encoded_traj_diffs_mean:", encoded_traj_diffs_mean)
    print("encoded_langs_mean:", encoded_langs_mean)
    print("dot_prods_mean:", dot_prods_mean)
    print("pos_dot_prods_mean:", pos_dot_prods_mean)
    print("neg_dot_prods_mean:", neg_dot_prods_mean)

    print("encoded_traj_diffs_std:", encoded_traj_diffs_std)
    print("encoded_langs_std:", encoded_langs_std)
    print("dot_prods_std:", dot_prods_std)
    print("pos_dot_prods_std:", pos_dot_prods_std)
    print("neg_dot_prods_std:", neg_dot_prods_std)


def add_embeddings(model, device, trajectories, reference_traj, nl_embedding):
    # reference_policy_obs = np.load(os.path.join(reference_policy_dir, "traj_observations.npy"))
    # reference_policy_act = np.load(os.path.join(reference_policy_dir, "traj_actions.npy"))
    # reference_policy_trajs = np.concatenate((reference_policy_obs, reference_policy_act), axis=-1)
    # reference_policy_traj = reference_policy_trajs[0]  # We just take the first rollout as our reference.
    #
    # comp_str = "Move faster."
    # bert_embedding = np.load('/home/jeremy/robosuite-benchmark/data/nl-traj/all-pairs/nlcomps.npy')[2]  # "Move faster is the 3rd string in file.

    reference_traj = torch.unsqueeze(torch.as_tensor(reference_traj, dtype=torch.float32, device=device), 0)
    nl_embedding = torch.unsqueeze(torch.as_tensor(nl_embedding, dtype=torch.float32, device=device), 0)
    encoded_ref_traj, _, encoded_comp_str, _, _ = model((reference_traj, reference_traj, nl_embedding))

    # This is the traj we are looking for.
    encoded_target_traj = encoded_ref_traj + encoded_comp_str

    max_cos_similarity = -1
    max_log_likelihood = -1e-5
    max_cos_similarity_traj = None
    max_log_likelihood_traj = None

    max_sim_policy = ''
    logsigmoid = nn.LogSigmoid()

    with torch.no_grad():
        for i in range(trajectories.shape[0]):
            traj = torch.unsqueeze(torch.as_tensor(trajectories[i, :, :], dtype=torch.float32, device=device), 0)
            encoded_traj, _, _, _, _ = model((traj, traj, nl_embedding))

            cos_similarity = F.cosine_similarity(encoded_target_traj, encoded_traj).item()
            dot_prod = torch.einsum('ij,ij->i', encoded_target_traj, encoded_traj)
            log_likelihood = logsigmoid(dot_prod).item()
            if cos_similarity > max_cos_similarity:
                max_cos_similarity = cos_similarity
                max_cos_similarity_traj = traj.detach().cpu().numpy()
            if log_likelihood > max_log_likelihood:
                max_log_likelihood = log_likelihood
                max_log_likelihood_traj = traj.detach().cpu().numpy()

    ### NOTE: BELOW CONTAINS THE OLD IMPLEMENTATION
    # for config in os.listdir(policy_dir):
    #     policy_path = os.path.join(policy_dir, config)
    #     if os.path.isdir(policy_path) and os.listdir(policy_path):  # Check that policy_path is a directory and that directory is not empty
    #         # print(policy_path)
    #         observations = np.load(os.path.join(policy_path, "traj_observations.npy"))
    #         actions = np.load(os.path.join(policy_path, "traj_actions.npy"))
    #         rewards = np.load(os.path.join(policy_path, "traj_rewards.npy"))
    #         # observations has dimensions (n_trajs, n_timesteps, obs_dimension)
    #         trajs = np.concatenate((observations, actions), axis=-1)
    #
    #         # Downsample
    #         trajs = trajs[0:trajs_per_policy]
    #         rewards = rewards[0:trajs_per_policy]
    #
    #         trajs = torch.as_tensor(trajs, dtype=torch.float32)
    #         encoded_traj, _, _, _, _ = model((trajs, trajs, bert_embedding))
    #
    #         for i in range(trajs_per_policy):
    #             similarity = F.cosine_similarity(encoded_target_traj, encoded_traj[i])
    #             similarity = similarity.item()
    #             if similarity > max_similarity:
    #                 max_similarity = similarity
    #                 max_sim_policy = policy_path + ' ' + str(i)
    #                 print("max sim so far at:", policy_path)
    #                 print("i:", i)

    print("max cos similarity:", max_cos_similarity)
    print("max_log_likelihood:", max_log_likelihood)

    return max_cos_similarity_traj, max_log_likelihood_traj


def run_accuracy_check(model, device, n_trajs, trajectories, nl_comps, nl_embeddings, similarity_metric='log_likelihood'):
    p = np.random.permutation(n_trajs)
    trajectories = trajectories[p]
    ref_trajs = trajectories[0:n_trajs]

    num_correct = 0

    for ref_traj in ref_trajs:
        for nl_comp, nl_embedding in zip(nl_comps, nl_embeddings):
            max_cos_similarity_traj, max_log_likelihood_traj = add_embeddings(model, device, trajectories, ref_traj, nl_embedding)
            if similarity_metric == 'log_likelihood':
                target_traj = max_log_likelihood_traj
            else:
                target_traj = max_cos_similarity_traj

            # Greater
            if len([adj for adj in robosuite.synthetic_comparisons.greater_speed_adjs if adj in nl_comp]) > 0:
                ref_traj_feature_values = [speed(ref_traj[t]) for t in range(len(ref_traj))]
                print("nl_comp:", nl_comp)
                print("ref_traj speed:", np.mean(ref_traj_feature_values))

                target_traj_feature_values = [speed(target_traj[t]) for t in range(len(target_traj))]
                print("target_traj speed:", np.mean(target_traj_feature_values))
                if np.mean(target_traj_feature_values) > np.mean(ref_traj_feature_values):
                    num_correct += 1

            elif len([adj for adj in robosuite.synthetic_comparisons.greater_height_adjs if adj in nl_comp]) > 0:
                ref_traj_feature_values = [height(ref_traj[t]) for t in range(len(ref_traj))]
                target_traj_feature_values = [height(target_traj[t]) for t in range(len(target_traj))]
                if np.mean(target_traj_feature_values) > np.mean(ref_traj_feature_values):
                    num_correct += 1

            elif len([adj for adj in robosuite.synthetic_comparisons.greater_distance_adjs if adj in nl_comp]) > 0 and "bottle" in nl_comp:
                ref_traj_feature_values = [distance_to_bottle(ref_traj[t]) for t in range(len(ref_traj))]
                target_traj_feature_values = [distance_to_bottle(target_traj[t]) for t in range(len(target_traj))]
                if np.mean(target_traj_feature_values) > np.mean(ref_traj_feature_values):
                    num_correct += 1

            elif len([adj for adj in robosuite.synthetic_comparisons.greater_distance_adjs if adj in nl_comp]) > 0 and "cube" in nl_comp:
                ref_traj_feature_values = [distance_to_cube(ref_traj[t]) for t in range(len(ref_traj))]
                target_traj_feature_values = [distance_to_cube(target_traj[t]) for t in range(len(target_traj))]
                if np.mean(target_traj_feature_values) > np.mean(ref_traj_feature_values):
                    num_correct += 1

            # Lesser
            elif len([adj for adj in robosuite.synthetic_comparisons.less_speed_adjs if adj in nl_comp]) > 0:
                ref_traj_feature_values = [speed(ref_traj[t]) for t in range(len(ref_traj))]
                target_traj_feature_values = [speed(target_traj[t]) for t in range(len(target_traj))]
                if np.mean(target_traj_feature_values) < np.mean(ref_traj_feature_values):
                    num_correct += 1

            elif len([adj for adj in robosuite.synthetic_comparisons.less_height_adjs if adj in nl_comp]) > 0:
                ref_traj_feature_values = [height(ref_traj[t]) for t in range(len(ref_traj))]
                target_traj_feature_values = [height(target_traj[t]) for t in range(len(target_traj))]
                if np.mean(target_traj_feature_values) < np.mean(ref_traj_feature_values):
                    num_correct += 1

            elif len([adj for adj in robosuite.synthetic_comparisons.less_distance_adjs if adj in nl_comp]) > 0 and "bottle" in nl_comp:
                ref_traj_feature_values = [distance_to_bottle(ref_traj[t]) for t in range(len(ref_traj))]
                target_traj_feature_values = [distance_to_bottle(target_traj[t]) for t in range(len(target_traj))]
                if np.mean(target_traj_feature_values) < np.mean(ref_traj_feature_values):
                    num_correct += 1

            elif len([adj for adj in robosuite.synthetic_comparisons.less_distance_adjs if adj in nl_comp]) > 0 and "cube" in nl_comp:
                ref_traj_feature_values = [distance_to_cube(ref_traj[t]) for t in range(len(ref_traj))]
                target_traj_feature_values = [distance_to_cube(target_traj[t]) for t in range(len(target_traj))]
                if np.mean(target_traj_feature_values) < np.mean(ref_traj_feature_values):
                    num_correct += 1

    print("num_correct:", num_correct)
    print("accuracy:", num_correct / (n_trajs * len(nl_comps)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--seed', type=int, default=0, help='')
    parser.add_argument('--model-path', type=str, default='', help='')
    parser.add_argument('--data-dir', type=str, default='', help='')
    parser.add_argument('--val', action="store_true", help='')
    parser.add_argument('--n-trajs', type=int, default=0, help='')
    parser.add_argument('--similarity-metric', type=str, default='', help='')

    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    model_path = args.model_path
    data_dir = args.data_dir
    val = args.val  # Whether or not to use validation (default train)
    n_trajs = args.n_trajs
    similarity_metric = args.similarity_metric

    model, device = load_model(model_path)
    if val:
        nl_comp_file = os.path.join(data_dir, "val/nlcomps.json")
        traj_file = os.path.join(data_dir, "val/trajs.npy")
    else:
        nl_comp_file = os.path.join(data_dir, "train/nlcomps.json")
        traj_file = os.path.join(data_dir, "train/trajs.npy")

    with open(nl_comp_file, 'rb') as f:
        nl_comps = json.load(f)
    trajs = np.load(traj_file)

    # IMPORTANT: Make this a unique set of nl comps
    nl_comps = list(set(nl_comps))
    nl_embeddings = preprocess_strings('', 500, nl_comps)

    run_accuracy_check(model, device, n_trajs, trajs, nl_comps, nl_embeddings, similarity_metric)




