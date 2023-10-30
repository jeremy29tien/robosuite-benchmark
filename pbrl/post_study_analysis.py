from pbrl.run_aprel import make_gym_env, load_model, calc_and_set_global_vars, ENCODED_TRAJECTORIES_MEAN, ENCODED_TRAJECTORIES_STD
import aprel
import numpy as np

import torch
from nl_traj_feature_learning.learn_features import NLTrajAutoencoder
from nl_traj_feature_learning.nl_traj_dataset import NLTrajComparisonDataset
from nl_traj_feature_learning.learn_features import STATE_DIM, ACTION_DIM, BERT_OUTPUT_DIM

from robosuite.environments.manipulation.lift_features import gt_reward, speed, height, distance_to_bottle, distance_to_cube

import argparse
import os
import pickle
import json


user_results_dir = os.path.join('robosuite-benchmark/pbrl/results')
nlcommand_output_dir = os.path.join(user_results_dir,
                                    '128hidden_expertx50_noise-augmentation10_0.001weightdecay_simuserbeta500_nlcommand_mutualinformation_normalizefeaturefuncs_seed2')
preference_output_dir = os.path.join(user_results_dir,
                                     '128hidden_expertx50_noise-augmentation10_0.001weightdecay_simuserbeta500_mutualinformation_normalizefeaturefuncs_seed2')
model_path = 'robosuite-benchmark/models/fixed/128hidden_expertx50_noise-augmentation10_0.001weightdecay/model.pth'
traj_dir = 'robosuite-benchmark/data/nl-traj/56x3_expertx50_all-pairs_noise-augmentation10_id-mapping_with-videos_seed251'
video_dir = 'robosuite-benchmark/data/diverse-rewards/videos-very-short'
args = dict()
args['normalize_feature_funcs'] = True
seed = 2
human_user = False
args["sim_user_beta"] = 500

gym_env = make_gym_env(seed)

encoder_model, device = load_model(model_path)

if args['normalize_feature_funcs']:
    trajs = np.concatenate(
        (np.load(os.path.join(traj_dir, 'train/trajs.npy')), np.load(os.path.join(traj_dir, 'val/trajs.npy'))), axis=0)
    calc_and_set_global_vars(trajs, encoder_model, device)
    from pbrl.run_aprel import GT_REWARD_MEAN, GT_REWARD_STD, SPEED_MEAN, SPEED_STD, HEIGHT_MEAN, HEIGHT_STD, \
        DISTANCE_TO_BOTTLE_MEAN, DISTANCE_TO_BOTTLE_STD, DISTANCE_TO_CUBE_MEAN, DISTANCE_TO_CUBE_STD, \
        ENCODED_TRAJECTORIES_MEAN, ENCODED_TRAJECTORIES_STD
    # print(ENCODED_TRAJECTORIES_MEAN)


def feature_func(traj):
    """Returns the features of the given trajectory, i.e. \Phi(traj).

    Args:
        traj: List of state-action tuples, e.g. [(state0, action0), (state1, action1), ...]

    Returns:
        features: a numpy vector corresponding the features of the trajectory
    """
    traj = np.asarray([np.concatenate((t[0], t[1]), axis=0) for t in traj if t[1] is not None and t[0] is not None])
    traj = torch.unsqueeze(torch.as_tensor(traj, dtype=torch.float32, device=device), 0)
    rand_traj = torch.rand(traj.shape, device=device)
    rand_nl = torch.rand(1, BERT_OUTPUT_DIM, device=device)
    with torch.no_grad():
        encoded_traj, _, _, _, _ = encoder_model((traj, rand_traj, rand_nl))
        encoded_traj = encoded_traj.squeeze().detach().cpu().numpy()

    # Normalize embedding; check that the normalization corresponds to the dataset/model.
    if args['normalize_feature_funcs']:
        encoded_traj = (encoded_traj - ENCODED_TRAJECTORIES_MEAN) / ENCODED_TRAJECTORIES_STD
    return encoded_traj


env = aprel.Environment(gym_env, feature_func)

# Take trajectories from our training/val data.
train_trajs = np.load(os.path.join(traj_dir, 'train/trajs.npy'))
val_trajs = np.load(os.path.join(traj_dir, 'val/trajs.npy'))
videos_available = True
try:
    train_traj_video_ids = np.load(os.path.join(traj_dir, 'train/traj_video_ids.npy'))
    val_traj_video_ids = np.load(os.path.join(traj_dir, 'val/traj_video_ids.npy'))
except FileNotFoundError:
    print("No trajectory videos found, proceeding without them.")
    videos_available = False

train_traj_set = []
for i, train_traj in enumerate(train_trajs):
    # If available, load trajectories with camera observation.
    clip_path = None
    if videos_available:
        clip_path = os.path.join(video_dir, str(train_traj_video_ids[i]) + ".mp4")

    traj = aprel.Trajectory(env, [(t[0:STATE_DIM], t[STATE_DIM:STATE_DIM + ACTION_DIM]) for t in train_traj],
                            clip_path=clip_path)
    train_traj_set.append(traj)

val_traj_set = []
for i, val_traj in enumerate(val_trajs):
    # If available, load trajectories with camera observation.
    clip_path = None
    if videos_available:
        clip_path = os.path.join(video_dir, str(val_traj_video_ids[i]) + ".mp4")

    traj = aprel.Trajectory(env, [(t[0:STATE_DIM], t[STATE_DIM:STATE_DIM + ACTION_DIM]) for t in val_traj],
                            clip_path=clip_path)
    val_traj_set.append(traj)

trajectory_set = aprel.TrajectorySet(train_traj_set)
val_trajectory_set = aprel.TrajectorySet(val_traj_set)

# Load val_data
if human_user:
    with open(os.path.join(user_results_dir, 'val_data.pkl'), 'rb') as f:
        val_data = pickle.load(f)
else:
    def true_user_feature_func(traj):
        """Returns the features of the given trajectory, i.e. \Phi(traj).

        Args:
            traj: List of state-action tuples, e.g. [(state0, action0), (state1, action1), ...]

        Returns:
            features: a numpy vector corresponding the features of the trajectory
        """
        traj = np.asarray([np.concatenate((t[0], t[1]), axis=0) for t in traj if t[1] is not None and t[0] is not None])
        features = np.zeros(5)
        features[0] = np.mean([gt_reward(t) for t in traj])
        features[1] = np.mean([speed(t) for t in traj])
        features[2] = np.mean([height(t) for t in traj])
        features[3] = np.mean([distance_to_bottle(t) for t in traj])
        features[4] = np.mean([distance_to_cube(t) for t in traj])

        if args['normalize_feature_funcs']:
            features[0] = (features[0] - GT_REWARD_MEAN) / GT_REWARD_STD
            features[1] = (features[1] - SPEED_MEAN) / SPEED_STD
            features[2] = (features[2] - HEIGHT_MEAN) / HEIGHT_STD
            features[3] = (features[3] - DISTANCE_TO_BOTTLE_MEAN) / DISTANCE_TO_BOTTLE_STD
            features[4] = (features[4] - DISTANCE_TO_CUBE_MEAN) / DISTANCE_TO_CUBE_STD

        return features
    val_data = []
    true_params = {'weights': np.array([1, 0, 0, 0, 0]),
                   'beta': args['sim_user_beta'],
                   'feature_func': true_user_feature_func,
                   'trajectory_set': trajectory_set}
    true_user = aprel.CustomFeatureUser(true_params)

    for i in range(val_trajectory_set.size):
        for j in range(i + 1, val_trajectory_set.size):
            # Log likelihood under learned reward
            val_query = aprel.PreferenceQuery([val_trajectory_set[i], val_trajectory_set[j]])
            val_response = true_user.respond(val_query)

            data = aprel.Preference(val_query, val_response[0])
            val_data.append(data)

    # Create NLCommand val data point
    nl_comp_file = os.path.join(traj_dir, "train/unique_nlcomps_for_aprel.json")
    with open(nl_comp_file, 'rb') as f:
        nl_comps = json.load(f)

    nl_embedding_file = os.path.join(traj_dir, "train/unique_nlcomps_for_aprel.npy")
    nl_embeddings = np.load(nl_embedding_file)
    assert len(nl_comps) == len(nl_embeddings)

    global lang_encoder_func

    def lang_encoder_func(in_str: str) -> np.array:
        """Returns encoded version of in_str, i.e. \Phi(in_str).

        Args:
            in_str: Command in natural language.

        Returns:
            enc_str: a numpy vector corresponding the encoded string
        """
        enc_str = None
        for i in range(len(nl_comps)):
            if nl_comps[i] == in_str:
                enc_str = nl_embeddings[i]
                break
        if enc_str is None:
            raise ValueError("in_str must be a valid string, was instead:" + in_str)

        # Encode BERT-preprocessed string using learned model
        enc_str = torch.unsqueeze(torch.as_tensor(enc_str, dtype=torch.float32, device=device), 0)
        rand_traj = torch.rand(1, 500, STATE_DIM + ACTION_DIM, device=device)
        with torch.no_grad():
            _, _, enc_str, _, _ = encoder_model((rand_traj, rand_traj, enc_str))
            enc_str = enc_str.squeeze().detach().cpu().numpy()

        return enc_str

    query = aprel.NLCommandQuery(trajectory_set[:1], lang_encoder_func, nl_comps, nl_embeddings)
    for i in range(val_trajectory_set.size):
        val_query = query.copy()
        val_query.slate = [val_trajectory_set[i]]
        val_response = true_user.respond(val_query)

        data = aprel.NLCommand(val_query, val_response[0])
        val_data.append(data)


def eval_treatment(output_dir):
    weights_per_iter = np.load(os.path.join(output_dir, 'weights_per_iter.npy'))
    val_lls_per_iter = []
    for weights in weights_per_iter:
        latest_params = {'weights': weights,
                         'trajectory_set': trajectory_set}
        eval_user_model = aprel.SoftmaxUser(latest_params)

        val_lls = []
        for data in val_data:
            ll = eval_user_model.loglikelihood(data)
            val_lls.append(ll)
        val_lls_per_iter.append(np.mean(val_lls))
        np.save(os.path.join(output_dir, 'user_study_val_log_likelihoods.npy'), val_lls_per_iter)
        print("Val log likelihoods per iteration:", val_lls_per_iter)


eval_treatment(nlcommand_output_dir)
eval_treatment(preference_output_dir)
