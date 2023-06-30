import torch
from torch.utils.data import Dataset
import json
import numpy as np
import argparse
import os
import torch.nn as nn
import torch.nn.functional as F
from nl_traj_feature_learning.learn_features import NLTrajAutoencoder, STATE_DIM, ACTION_DIM, BERT_OUTPUT_DIM
from nl_traj_feature_learning.nl_traj_dataset import NLTrajComparisonDataset
from gpu_utils import determine_default_torch_device
import robosuite.synthetic_comparisons
from robosuite.environments.manipulation.lift_features import gt_reward, speed, height, distance_to_bottle, \
    distance_to_cube
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
    try:
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
    except FileNotFoundError:
        train_nlcomp_index_file = os.path.join(data_dir, "train/nlcomp_indexes.npy")
        train_unique_nlcomp_file = os.path.join(data_dir, "train/unique_nlcomps.npy")
        val_nlcomp_index_file = os.path.join(data_dir, "val/nlcomp_indexes.npy")
        val_unique_nlcomp_file = os.path.join(data_dir, "val/unique_nlcomps.npy")

        train_traj_a_index_file = os.path.join(data_dir, "train/traj_a_indexes.npy")
        train_traj_b_index_file = os.path.join(data_dir, "train/traj_b_indexes.npy")
        train_traj_file = os.path.join(data_dir, "train/trajs.npy")
        val_traj_a_index_file = os.path.join(data_dir, "val/traj_a_indexes.npy")
        val_traj_b_index_file = os.path.join(data_dir, "val/traj_b_indexes.npy")
        val_traj_file = os.path.join(data_dir, "val/trajs.npy")

        train_dataset = NLTrajComparisonDataset(train_nlcomp_index_file, train_traj_a_index_file,
                                                train_traj_b_index_file,
                                                preprocessed_nlcomps=True, id_mapped=True,
                                                unique_nlcomp_file=train_unique_nlcomp_file, traj_file=train_traj_file)
        val_dataset = NLTrajComparisonDataset(val_nlcomp_index_file, val_traj_a_index_file, val_traj_b_index_file,
                                              preprocessed_nlcomps=True, id_mapped=True,
                                              unique_nlcomp_file=val_unique_nlcomp_file, traj_file=val_traj_file)

    # Statistics that have to do with the trajectories
    all_trajectories = np.concatenate((train_dataset.trajs, val_dataset.trajs), axis=0)
    all_encoded_trajectories = []
    for traj in all_trajectories:
        traj = torch.unsqueeze(torch.as_tensor(traj, dtype=torch.float32, device=device), 0)
        rand_traj = torch.rand(traj.shape, device=device)
        rand_nl = torch.rand(1, BERT_OUTPUT_DIM, device=device)
        with torch.no_grad():
            encoded_traj, _, _, _, _ = model((traj, rand_traj, rand_nl))
            encoded_traj = encoded_traj.squeeze().detach().cpu().numpy()

        all_encoded_trajectories.append(encoded_traj)

    all_encoded_trajectories_mean = np.mean(all_encoded_trajectories, axis=0)
    all_encoded_trajectories_std = np.std(all_encoded_trajectories, axis=0)
    print("all_encoded_trajectories_mean:", all_encoded_trajectories_mean)
    print("all_encoded_trajectories_std:", all_encoded_trajectories_std)

    # Statistics that have to do with the actual dataset
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


def find_closest_policy(model, device, policy_dir, reference_policy_dir, nl_comp, nl_embedding, similarity_metric,
                        trajs_per_policy=3):
    reference_policy_obs = np.load(os.path.join(reference_policy_dir, "traj_observations.npy"))
    reference_policy_act = np.load(os.path.join(reference_policy_dir, "traj_actions.npy"))
    reference_policy_trajs = np.concatenate((reference_policy_obs, reference_policy_act), axis=-1)
    reference_traj = reference_policy_trajs[0]  # We just take the first rollout as our reference.

    reference_traj = torch.unsqueeze(torch.as_tensor(reference_traj, dtype=torch.float32, device=device), 0)
    nl_embedding = torch.unsqueeze(torch.as_tensor(nl_embedding, dtype=torch.float32, device=device), 0)
    with torch.no_grad():
        encoded_ref_traj, _, encoded_comp_str, _, _ = model((reference_traj, reference_traj, nl_embedding))

    # This is the traj we are looking for.
    encoded_target_traj = encoded_ref_traj + encoded_comp_str

    print("encoded_ref_traj:", encoded_ref_traj)
    print("encoded_comp_str:", encoded_comp_str)
    print("encoded_target_traj:", encoded_target_traj)

    if similarity_metric == 'cos_similarity':
        max_sim_metric = -1
    else:
        max_sim_metric = -1e-5
    max_sim_policy = ''

    logsigmoid = nn.LogSigmoid()

    for config in os.listdir(policy_dir):
        policy_path = os.path.join(policy_dir, config)
        if os.path.isdir(policy_path) and os.listdir(
                policy_path):  # Check that policy_path is a directory and that directory is not empty
            # print(policy_path)
            observations = np.load(os.path.join(policy_path, "traj_observations.npy"))
            actions = np.load(os.path.join(policy_path, "traj_actions.npy"))
            rewards = np.load(os.path.join(policy_path, "traj_rewards.npy"))
            # observations has dimensions (n_trajs, n_timesteps, obs_dimension)
            trajs = np.concatenate((observations, actions), axis=-1)

            # Downsample
            trajs = trajs[0:trajs_per_policy]
            rewards = rewards[0:trajs_per_policy]

            trajs = torch.as_tensor(trajs, dtype=torch.float32, device=device)
            encoded_trajs, _, _, _, _ = model((trajs, trajs, nl_embedding))

            for i in range(trajs_per_policy):

                if similarity_metric == 'cos_similarity':
                    cos_similarity = F.cosine_similarity(encoded_comp_str, encoded_trajs[i] - encoded_ref_traj).item()
                    if cos_similarity > max_sim_metric:
                        # print("encoded_traj:", encoded_traj)
                        # print("encoded_traj - encoded_ref_traj:", encoded_traj - encoded_ref_traj)
                        # print("cos_similarity:", cos_similarity)
                        max_sim_metric = cos_similarity
                        max_sim_policy = 'policy: ' + policy_path + '\nrollout: ' + str(i)

                elif similarity_metric == 'log_likelihood':
                    dot_prod = torch.einsum('ij,ij->i', encoded_target_traj, encoded_trajs[i])
                    log_likelihood = logsigmoid(dot_prod).item()
                    if log_likelihood > max_sim_metric:
                        # print("encoded_traj:", encoded_traj)
                        max_sim_metric = log_likelihood
                        max_sim_policy = 'policy: ' + policy_path + '\nrollout: ' + str(i)
                else:
                    raise NotImplementedError('That similarity metric is not supported yet :(')

    print("max_sim_policy:", max_sim_policy)
    print("max " + similarity_metric + ":", max_sim_metric)
    return max_sim_policy, max_sim_metric


def add_embeddings(model, device, trajectories, reference_traj, nl_embedding, similarity_metric):
    reference_traj = torch.unsqueeze(torch.as_tensor(reference_traj, dtype=torch.float32, device=device), 0)
    nl_embedding = torch.unsqueeze(torch.as_tensor(nl_embedding, dtype=torch.float32, device=device), 0)
    with torch.no_grad():
        encoded_ref_traj, _, encoded_comp_str, _, _ = model((reference_traj, reference_traj, nl_embedding))

    # print("reference_traj:", reference_traj)
    print("encoded_ref_traj:", encoded_ref_traj)
    print("encoded_comp_str:", encoded_comp_str)

    # This is the traj we are looking for.
    encoded_target_traj = encoded_ref_traj + encoded_comp_str

    print("encoded_target_traj:", encoded_target_traj)

    if similarity_metric == 'cos_similarity':
        max_sim_metric = -1
    else:
        max_sim_metric = -1e-5
    max_sim_traj = None
    encoded_max_sim_traj = None

    logsigmoid = nn.LogSigmoid()

    with torch.no_grad():
        for i in range(trajectories.shape[0]):
            traj = torch.unsqueeze(torch.as_tensor(trajectories[i, :, :], dtype=torch.float32, device=device), 0)
            encoded_traj, _, _, _, _ = model((traj, traj, nl_embedding))

            if similarity_metric == 'cos_similarity':
                cos_similarity = F.cosine_similarity(encoded_comp_str, encoded_traj - encoded_ref_traj).item()
                if cos_similarity > max_sim_metric:
                    # print("encoded_traj:", encoded_traj)
                    # print("encoded_traj - encoded_ref_traj:", encoded_traj - encoded_ref_traj)
                    # print("cos_similarity:", cos_similarity)
                    max_sim_metric = cos_similarity
                    max_sim_traj = traj.squeeze().detach().cpu().numpy()
                    encoded_max_sim_traj = encoded_traj.squeeze().detach().cpu().numpy()

            elif similarity_metric == 'log_likelihood':
                dot_prod = torch.einsum('ij,ij->i', encoded_target_traj, encoded_traj)
                log_likelihood = logsigmoid(dot_prod).item()
                if log_likelihood > max_sim_metric:
                    # print("encoded_traj:", encoded_traj)
                    max_sim_metric = log_likelihood
                    max_sim_traj = traj.squeeze().detach().cpu().numpy()
                    encoded_max_sim_traj = encoded_traj.squeeze().detach().cpu().numpy()

            else:
                raise NotImplementedError('That similarity metric is not supported yet :(')

    # print("max_sim_traj:", max_sim_traj)
    print("encoded_max_sim_traj:", encoded_max_sim_traj)
    print("max " + similarity_metric + ":", max_sim_metric)
    return max_sim_traj, max_sim_metric


def run_accuracy_check(model, device, n_trajs, trajectories, nl_comps, nl_embeddings,
                       similarity_metric='log_likelihood'):
    p = np.random.permutation(n_trajs)
    trajectories = trajectories[p]
    ref_trajs = trajectories[0:n_trajs]

    num_correct = 0
    num_incorrect = 0
    max_similarities = []

    for ref_traj in ref_trajs:
        for nl_comp, nl_embedding in zip(nl_comps, nl_embeddings):
            print("Adding embedding for", nl_comp)
            target_traj, max_similarity = add_embeddings(model, device, trajectories, ref_traj, nl_embedding,
                                                         similarity_metric)
            max_similarities.append(max_similarity)
            # Greater
            if len([adj for adj in robosuite.synthetic_comparisons.greater_gtreward_adjs if adj in nl_comp]) > 0:
                ref_traj_feature_values = [gt_reward(ref_traj[t]) for t in range(len(ref_traj))]
                target_traj_feature_values = [gt_reward(target_traj[t]) for t in range(len(target_traj))]
                print("ref_traj gt_reward:", np.mean(ref_traj_feature_values))
                print("target_traj gt_reward:", np.mean(target_traj_feature_values))
                if np.mean(target_traj_feature_values) > np.mean(ref_traj_feature_values):
                    num_correct += 1
                    print("GT reward is indeed greater.")
                else:
                    num_incorrect += 1
                    print("GT reward is actually lesser.")

            elif len([adj for adj in robosuite.synthetic_comparisons.greater_speed_adjs if adj in nl_comp]) > 0:
                ref_traj_feature_values = [speed(ref_traj[t]) for t in range(len(ref_traj))]
                target_traj_feature_values = [speed(target_traj[t]) for t in range(len(target_traj))]
                # print("ref_traj.shape", np.asarray(ref_traj).shape)
                # print("target_traj.shape:", np.asarray(target_traj).shape)

                # print("nl_comp:", nl_comp)
                # print("nl_embedding:", nl_embedding)
                print("ref_traj speed:", np.mean(ref_traj_feature_values))
                print("target_traj speed:", np.mean(target_traj_feature_values))
                if np.mean(target_traj_feature_values) > np.mean(ref_traj_feature_values):
                    num_correct += 1
                    print("Speed is indeed greater.")
                else:
                    num_incorrect += 1
                    print("Speed is actually lesser.")

            elif len([adj for adj in robosuite.synthetic_comparisons.greater_height_adjs if adj in nl_comp]) > 0:
                ref_traj_feature_values = [height(ref_traj[t]) for t in range(len(ref_traj))]
                target_traj_feature_values = [height(target_traj[t]) for t in range(len(target_traj))]

                # print("nl_comp:", nl_comp)
                # print("nl_embedding:", nl_embedding)
                print("ref_traj height:", np.mean(ref_traj_feature_values))
                print("target_traj height:", np.mean(target_traj_feature_values))

                if np.mean(target_traj_feature_values) > np.mean(ref_traj_feature_values):
                    num_correct += 1
                    print("Height is indeed greater.")
                else:
                    num_incorrect += 1
                    print("Height is actually lesser.")

            elif len([adj for adj in robosuite.synthetic_comparisons.greater_distance_adjs if
                      adj in nl_comp]) > 0 and "bottle" in nl_comp:
                ref_traj_feature_values = [distance_to_bottle(ref_traj[t]) for t in range(len(ref_traj))]
                target_traj_feature_values = [distance_to_bottle(target_traj[t]) for t in range(len(target_traj))]
                print("ref_traj distance from bottle:", np.mean(ref_traj_feature_values))
                print("target_traj distance from bottle:", np.mean(target_traj_feature_values))
                if np.mean(target_traj_feature_values) > np.mean(ref_traj_feature_values):
                    num_correct += 1
                    print("Distance is indeed greater.")
                else:
                    num_incorrect += 1
                    print("Distance is actually lesser.")

            elif len([adj for adj in robosuite.synthetic_comparisons.greater_distance_adjs if
                      adj in nl_comp]) > 0 and "cube" in nl_comp:
                ref_traj_feature_values = [distance_to_cube(ref_traj[t]) for t in range(len(ref_traj))]
                target_traj_feature_values = [distance_to_cube(target_traj[t]) for t in range(len(target_traj))]
                print("ref_traj distance from cube:", np.mean(ref_traj_feature_values))
                print("target_traj distance from cube:", np.mean(target_traj_feature_values))
                if np.mean(target_traj_feature_values) > np.mean(ref_traj_feature_values):
                    num_correct += 1
                    print("Distance is indeed greater.")
                else:
                    num_incorrect += 1
                    print("Distance is actually lesser.")

            # Lesser
            elif len([adj for adj in robosuite.synthetic_comparisons.less_gtreward_adjs if adj in nl_comp]) > 0:
                ref_traj_feature_values = [gt_reward(ref_traj[t]) for t in range(len(ref_traj))]
                target_traj_feature_values = [gt_reward(target_traj[t]) for t in range(len(target_traj))]
                print("ref_traj gt_reward:", np.mean(ref_traj_feature_values))
                print("target_traj gt_reward:", np.mean(target_traj_feature_values))
                if np.mean(target_traj_feature_values) < np.mean(ref_traj_feature_values):
                    num_correct += 1
                    print("GT reward is indeed lesser.")
                else:
                    num_incorrect += 1
                    print("GT reward is actually greater.")

            elif len([adj for adj in robosuite.synthetic_comparisons.less_speed_adjs if adj in nl_comp]) > 0:
                ref_traj_feature_values = [speed(ref_traj[t]) for t in range(len(ref_traj))]
                target_traj_feature_values = [speed(target_traj[t]) for t in range(len(target_traj))]
                # print("ref_traj.shape", np.asarray(ref_traj).shape)
                # print("target_traj.shape:", np.asarray(target_traj).shape)

                # print("nl_comp:", nl_comp)
                # print("nl_embedding:", nl_embedding)
                print("ref_traj speed:", np.mean(ref_traj_feature_values))
                print("target_traj speed:", np.mean(target_traj_feature_values))
                if np.mean(target_traj_feature_values) < np.mean(ref_traj_feature_values):
                    num_correct += 1
                    print("Speed is indeed lesser.")
                else:
                    num_incorrect += 1
                    print("Speed is actually greater.")

            elif len([adj for adj in robosuite.synthetic_comparisons.less_height_adjs if adj in nl_comp]) > 0:
                ref_traj_feature_values = [height(ref_traj[t]) for t in range(len(ref_traj))]
                target_traj_feature_values = [height(target_traj[t]) for t in range(len(target_traj))]

                # print("nl_comp:", nl_comp)
                # print("nl_embedding:", nl_embedding)
                print("ref_traj height:", np.mean(ref_traj_feature_values))
                print("target_traj height:", np.mean(target_traj_feature_values))

                if np.mean(target_traj_feature_values) < np.mean(ref_traj_feature_values):
                    num_correct += 1
                    print("Height is indeed lesser.")
                else:
                    num_incorrect += 1
                    print("Height is actually greater.")

            elif len([adj for adj in robosuite.synthetic_comparisons.less_distance_adjs if
                      adj in nl_comp]) > 0 and "bottle" in nl_comp:
                ref_traj_feature_values = [distance_to_bottle(ref_traj[t]) for t in range(len(ref_traj))]
                target_traj_feature_values = [distance_to_bottle(target_traj[t]) for t in range(len(target_traj))]
                print("ref_traj distance from bottle:", np.mean(ref_traj_feature_values))
                print("target_traj distance from bottle:", np.mean(target_traj_feature_values))
                if np.mean(target_traj_feature_values) < np.mean(ref_traj_feature_values):
                    num_correct += 1
                    print("Distance is indeed lesser.")
                else:
                    num_incorrect += 1
                    print("Distance is actually greater.")

            elif len([adj for adj in robosuite.synthetic_comparisons.less_distance_adjs if
                      adj in nl_comp]) > 0 and "cube" in nl_comp:
                ref_traj_feature_values = [distance_to_cube(ref_traj[t]) for t in range(len(ref_traj))]
                target_traj_feature_values = [distance_to_cube(target_traj[t]) for t in range(len(target_traj))]
                print("ref_traj distance from cube:", np.mean(ref_traj_feature_values))
                print("target_traj distance from cube:", np.mean(target_traj_feature_values))
                if np.mean(target_traj_feature_values) < np.mean(ref_traj_feature_values):
                    num_correct += 1
                    print("Distance is indeed lesser.")
                else:
                    num_incorrect += 1
                    print("Distance is actually greater.")
            else:
                print("THIS SHOULD NOT BE PRINTED.")
                raise ValueError("Unrecognized NL command.")
                # print("gt_reward nl_comp:", nl_comp)
            print('\n')

    print("num_correct:", num_correct)
    print("accuracy:", num_correct / (num_correct + num_incorrect))
    print("average max similarity:", np.mean(max_similarities))


def find_max_learned_reward(model, device, data_dir, reward_weights):
    train_trajectories = np.load(os.path.join(data_dir,
                                              "nl-traj/56x3_expertx50_all-pairs_noise-augmentation10_id-mapping_with-videos_seed251/train/trajs.npy"))
    val_trajectories = np.load(os.path.join(data_dir,
                                            "nl-traj/56x3_expertx50_all-pairs_noise-augmentation10_id-mapping_with-videos_seed251/val/trajs.npy"))
    trajectories = np.concatenate((train_trajectories, val_trajectories), axis=0)
    train_trajectory_video_ids = np.load(os.path.join(data_dir,
                                              "nl-traj/56x3_expertx50_all-pairs_noise-augmentation10_id-mapping_with-videos_seed251/train/traj_video_ids.npy.npy"))
    val_trajectory_video_ids = np.load(os.path.join(data_dir,
                                            "nl-traj/56x3_expertx50_all-pairs_noise-augmentation10_id-mapping_with-videos_seed251/val/traj_video_ids.npy.npy"))
    trajectory_video_ids = np.concatenate((train_trajectory_video_ids, val_trajectory_video_ids), axis=0)

    max_reward = -np.inf
    max_reward_traj_i = None
    max_reward_traj_video = None
    max_reward_traj = None
    for i, traj in enumerate(trajectories):
        traj = torch.unsqueeze(torch.as_tensor(traj, dtype=torch.float32, device=device), 0)
        rand_traj = torch.rand(traj.shape, device=device)
        rand_nl = torch.rand(1, BERT_OUTPUT_DIM, device=device)
        with torch.no_grad():
            encoded_traj, _, _, _, _ = model((traj, rand_traj, rand_nl))
            encoded_traj = encoded_traj.squeeze().detach().cpu().numpy()
        traj_reward = np.dot(reward_weights, encoded_traj)
        if traj_reward > max_reward:
            max_reward = traj_reward
            max_reward_traj_i = i
    max_reward_traj = trajectories[max_reward_traj_i]
    max_reward_traj_video = trajectory_video_ids[max_reward_traj_i]
    true_reward = [gt_reward(t) for t in max_reward_traj]
    print("Trajectory with highest returns:", max_reward_traj_video)
    print("Reward:", max_reward)
    print("True reward:", np.mean(true_reward))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--seed', type=int, default=0, help='')
    parser.add_argument('--model-path', type=str, default='', help='')
    parser.add_argument('--data-dir', type=str, default='', help='')
    parser.add_argument('--val', action="store_true", help='')
    parser.add_argument('--similarity-metric', type=str, default='', help='')

    # Arguments needed for --analyze
    parser.add_argument('--analyze', action="store_true", help='')
    parser.add_argument('--n-trajs', type=int, default=0, help='')

    # Arguments needed for --visualize
    parser.add_argument('--visualize', action="store_true", help='')
    parser.add_argument('--all-policy-dir', type=str, default='', help='')
    parser.add_argument('--reference-policy-dir', type=str, default='', help='')
    parser.add_argument('--command-string', type=str, default='', help='')

    # Arguments needed for --print-statistics
    parser.add_argument('--print-statistics', action="store_true", help='')

    # Arguments needed for --print-statistics
    parser.add_argument('--find-max-learned-reward', action="store_true", help='')

    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    model_path = args.model_path
    data_dir = args.data_dir
    val = args.val  # Whether or not to use validation (default train)
    similarity_metric = args.similarity_metric

    model, device = load_model(model_path)

    if args.analyze or args.visualize:
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

    if args.analyze:
        n_trajs = args.n_trajs
        run_accuracy_check(model, device, n_trajs, trajs, nl_comps, nl_embeddings, similarity_metric)
    elif args.visualize:
        policy_dir = args.all_policy_dir
        reference_policy_dir = args.reference_policy_dir
        nl_comp = args.command_string

        nl_embedding = None
        assert len(nl_comps) == len(nl_embeddings)
        for i in range(len(nl_comps)):
            if nl_comps[i] == nl_comp:
                nl_embedding = nl_embeddings[i]
                break
        if nl_embedding is None:
            raise ValueError("--command-string must be a valid string.")

        find_closest_policy(model, device, policy_dir, reference_policy_dir, nl_comp, nl_embedding, similarity_metric)
    elif args.print_statistics:
        print_embedding_statistics(model, device, data_dir, val)
    elif args.find_max_learned_reward:
        reward_weights = [0.91835815, 0.14147503, 0.12378615, 0.00524613, 0.13000946,
                          0.00349612, 0.08030579, 0.01395308, 0.13581616, -0.15866101,
                          -0.17884168, -0.01559023, -0.02973991, 0.03529254, 0.04609364,
                          0.13263787]
        find_max_learned_reward(model, device, data_dir, reward_weights)
    else:
        print("Need to specify either --analyze or --visualize or --print-statistics.")
