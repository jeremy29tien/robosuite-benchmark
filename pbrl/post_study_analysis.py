from pbrl.run_aprel import make_gym_env, load_model, calc_and_set_global_vars, ENCODED_TRAJECTORIES_MEAN, ENCODED_TRAJECTORIES_STD
import aprel
import numpy as np

import torch
from nl_traj_feature_learning.learn_features import NLTrajAutoencoder
from nl_traj_feature_learning.nl_traj_dataset import NLTrajComparisonDataset
from nl_traj_feature_learning.learn_features import STATE_DIM, ACTION_DIM, BERT_OUTPUT_DIM

import argparse
import os
import pickle

user_results_dir = os.path.join('robosuite-benchmark/pbrl/results/human_user/jtien')
nlcommand_output_dir = os.path.join(user_results_dir,
                                    '128hidden_expertx50_noise-augmentation10_0.001weightdecay_nlcommand_normalizefeaturefuncs')
preference_output_dir = os.path.join(user_results_dir,
                                     '128hidden_expertx50_noise-augmentation10_0.001weightdecay_preference_normalizefeaturefuncs')
model_path = 'robosuite-benchmark/models/fixed/128hidden_expertx50_noise-augmentation10_0.001weightdecay/model.pth'
traj_dir = 'robosuite-benchmark/data/nl-traj/56x3_expertx50_all-pairs_noise-augmentation10_id-mapping_with-videos_seed251'
video_dir = 'robosuite-benchmark/data/diverse-rewards/videos-very-short'
args = dict()
args['normalize_feature_funcs'] = True
seed = 0

gym_env = make_gym_env(seed)

encoder_model, device = load_model(model_path)

if args['normalize_feature_funcs']:
    trajs = np.concatenate(
        (np.load(os.path.join(traj_dir, 'train/trajs.npy')), np.load(os.path.join(traj_dir, 'val/trajs.npy'))), axis=0)
    calc_and_set_global_vars(trajs, encoder_model, device)
    from pbrl.run_aprel import ENCODED_TRAJECTORIES_MEAN, ENCODED_TRAJECTORIES_STD
    print(ENCODED_TRAJECTORIES_MEAN)


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
with open(os.path.join(user_results_dir, 'val_data.pkl'), 'rb') as f:
    val_data = pickle.load(f)


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
