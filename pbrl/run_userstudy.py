from pbrl.run_aprel import run_aprel, make_gym_env
import aprel
import numpy as np

import torch
from nl_traj_feature_learning.learn_features import NLTrajAutoencoder
from nl_traj_feature_learning.nl_traj_dataset import NLTrajComparisonDataset

import argparse
import os


def treatment_A():
    selection = None
    while selection is None:
        selection = input("Beginning treatment A, where you will be asked to provide feedback to the robot in the "
                          "form of a command in natural language. Ready? (type \'yes\' to proceed): ")
        if selection != 'yes':
            selection = None

    args['query_type'] = 'nl_command'
    args['query_size'] = 1
    gym_env = make_gym_env(seed)
    return run_aprel(seed, gym_env, model_path, human_user, traj_dir, video_dir,
                                                  nlcommand_output_dir, args)


def treatment_B():
    selection = None
    while selection is None:
        selection = input("Beginning treatment B, where you will be asked to provide feedback to the robot in the "
                          "form of a pairwise preference. Ready? (type \'yes\' to proceed): ")
        if selection != 'yes':
            selection = None

    args['query_type'] = 'preference'
    args['query_size'] = 2
    gym_env = make_gym_env(seed)
    return run_aprel(seed, gym_env, model_path, human_user, traj_dir, video_dir, preference_output_dir,
                                      args)


def eval_treatment_A():
    weights = np.load(os.path.join(nlcommand_output_dir, 'weights.npy'))
    latest_params = {'weights': weights,
                     'trajectory_set': trajectory_set}
    eval_user_model = aprel.SoftmaxUser(latest_params)

    val_lls = []
    for data in val_data:
        ll = eval_user_model.loglikelihood(data)
        val_lls.append(ll)
    print("Treatment A validation log likelihood:", np.mean(val_lls))
    np.save(os.path.join(nlcommand_output_dir, 'user_study_val_log_likelihood.npy'), np.mean(val_lls))


def eval_treatment_B():
    weights = np.load(os.path.join(preference_output_dir, 'weights.npy'))
    latest_params = {'weights': weights,
                     'trajectory_set': trajectory_set}
    eval_user_model = aprel.SoftmaxUser(latest_params)

    val_lls = []
    for data in val_data:
        ll = eval_user_model.loglikelihood(data)
        val_lls.append(ll)
    print("Treatment B validation log likelihood:", np.mean(val_lls))
    np.save(os.path.join(preference_output_dir, 'user_study_val_log_likelihood.npy'), np.mean(val_lls))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('--seed', type=int, default=0, help='')
    parser.add_argument('--model-path', type=str, default='robosuite-benchmark/models/fixed/128hidden_expertx50_noise-augmentation10_0.001weightdecay/model.pth', help='')
    parser.add_argument('--traj-dir', type=str, default='robosuite-benchmark/data/nl-traj/56x3_expertx50_all-pairs_noise-augmentation10_id-mapping_with-videos_seed251/', help='')
    parser.add_argument('--video-dir', type=str, default='robosuite-benchmark/data/diverse-rewards/videos-very-short', help='')
    # parser.add_argument('--output-dir', type=str, default='', help='')
    # parser.add_argument('--query_type', type=str, default='preference',
    #                     help='Type of the queries that will be actively asked to the user. Options: preference, nl_command, weak_comparison, full_ranking.')
    # parser.add_argument('--query_size', type=int, default=2,
    #                     help='Number of trajectories in each query.')
    parser.add_argument('--num_iterations', type=int, default=5,
                        help='Number of iterations in the active learning loop.')
    parser.add_argument('--optim_method', type=str, default='exhaustive_search',
                        help='Options: exhaustive_search, greedy, medoids, boundary_medoids, successive_elimination, dpp.')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size can be set >1 for batch active learning algorithms.')
    parser.add_argument('--acquisition', type=str, default='random',
                        help='Acquisition function for active querying. Options: mutual_information, volume_removal, disagreement, regret, random, thompson')
    parser.add_argument('--reduced_size_for_batches', type=int, default=100,
                        help='The number of greedily chosen candidate queries (reduced set) for batch generation.')
    parser.add_argument('--dpp_gamma', type=int, default=1,
                        help='Gamma parameter for the DPP method: the higher gamma the more important is the acquisition function relative to diversity.')
    # parser.add_argument('--normalize_feature_funcs', action="store_true", help='')
    parser.add_argument('--free_input', action="store_true", help='')
    parser.add_argument('--treatment_order', type=str, default='AB')
    parser.add_argument('--user_id', type=str, default='jtien')
    parser.add_argument('--verbose', action="store_true", help='')

    args = parser.parse_args()

    seed = args.seed
    model_path = args.model_path
    traj_dir = args.traj_dir
    video_dir = args.video_dir
    human_user = True
    args = vars(args)
    args['normalize_feature_funcs'] = True
    args['distance_metric_for_batches'] = aprel.default_query_distance

    user_results_dir = os.path.join('robosuite-benchmark/pbrl/results/human_user', args['user_id'])
    nlcommand_output_dir = os.path.join(user_results_dir, '128hidden_expertx50_noise-augmentation10_0.001weightdecay_nlcommand_normalizefeaturefuncs')
    preference_output_dir = os.path.join(user_results_dir, '128hidden_expertx50_noise-augmentation10_0.001weightdecay_preference_normalizefeaturefuncs')
    try:
        os.makedirs(nlcommand_output_dir, exist_ok=True)
    except OSError as error:
        print("Directory '%s' can not be created" % nlcommand_output_dir)
    try:
        os.makedirs(preference_output_dir, exist_ok=True)
    except OSError as error:
        print("Directory '%s' can not be created" % preference_output_dir)

    if args['treatment_order'] == 'AB':
        # Treatment A
        nlcommand_valdata, trajectory_set, nlcommand_best_traj = treatment_A()

        # Treatment B
        preference_valdata, _, preference_best_traj = treatment_B()
    else:
        # Treatment B
        preference_valdata, _, preference_best_traj = treatment_B()

        # Treatment A
        nlcommand_valdata, trajectory_set, nlcommand_best_traj = treatment_A()

    val_data = nlcommand_valdata + preference_valdata

    # Evaluate Treatment A
    eval_treatment_A()

    # Evaluate Treatment B
    eval_treatment_B()

    print("Final Question: Which trajectory is the best?")
    print("Playing trajectory #0")
    nlcommand_best_traj.visualize()
    print("Playing trajectory #1")
    preference_best_traj.visualize()
    answer = input("Enter a number: [0-1]: ")
    np.save(os.path.join(user_results_dir, 'final_answer.npy'), answer)


