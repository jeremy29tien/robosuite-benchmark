import aprel
import numpy as np
import gym

import robosuite as suite
from robosuite.wrappers import GymWrapper
from robosuite.controllers import load_controller_config, ALL_CONTROLLERS
from robosuite.environments.manipulation.lift_features import gt_reward, speed, height, distance_to_bottle, distance_to_cube

import torch
from nl_traj_feature_learning.learn_features import NLTrajAutoencoder
from nl_traj_feature_learning.nl_traj_dataset import NLTrajComparisonDataset
from nl_traj_feature_learning.learn_features import STATE_DIM, ACTION_DIM, BERT_OUTPUT_DIM

from gpu_utils import determine_default_torch_device
import argparse
import os
import json

from bert_preprocessing import preprocess_strings

# if args['normalize_feature_funcs'] and "128hidden_expertx50_noise-augmentation10_0.001weightdecay" in model_path and "56x3_expertx50_all-pairs_noise-augmentation10_id-mapping" in traj_dir:
# all_encoded_trajectories_mean = [-0.22682665, -1.2686706, 1.5685039, -1.2785618, 3.4092467, 0.0070117, 0.05656387, -0.01284434, -0.6828233 , -0.27281293, -0.56345856, 0.00778119, -0.12456893, 0.06893575, 0.00524702, 0.2000821]
# all_encoded_trajectories_std = [0.42180222, 1.0952618, 1.9824623, 1.7400833, 0.21317093, 0.01458023, 0.48504975, 0.0102681, 1.1641839, 1.3725176, 0.18389156, 0.01337132, 0.7205983, 1.0508065, 0.01209785, 0.9927857]
GT_REWARD_MEAN = None
GT_REWARD_STD = None
SPEED_MEAN = None
SPEED_STD = None
HEIGHT_MEAN = None
HEIGHT_STD = None
DISTANCE_TO_BOTTLE_MEAN = None
DISTANCE_TO_BOTTLE_STD = None
DISTANCE_TO_CUBE_MEAN = None
DISTANCE_TO_CUBE_STD = None
ENCODED_TRAJECTORIES_MEAN = None
ENCODED_TRAJECTORIES_STD = None


def calc_and_set_global_vars(trajs, model, device):
    horizon = len(trajs[0])
    avg_gt_rewards = []
    avg_speeds = []
    avg_heights = []
    avg_distance_to_bottles = []
    avg_distance_to_cubes = []
    all_encoded_trajectories = []

    for traj in trajs:
        avg_gt_rewards.append(np.mean([gt_reward(traj[t]) for t in range(horizon)]))
        avg_speeds.append(np.mean([speed(traj[t]) for t in range(horizon)]))
        avg_heights.append(np.mean([height(traj[t]) for t in range(horizon)]))
        avg_distance_to_bottles.append(np.mean([distance_to_bottle(traj[t]) for t in range(horizon)]))
        avg_distance_to_cubes.append(np.mean([distance_to_cube(traj[t]) for t in range(horizon)]))

        traj = torch.unsqueeze(torch.as_tensor(traj, dtype=torch.float32, device=device), 0)
        rand_traj = torch.rand(traj.shape, device=device)
        rand_nl = torch.rand(1, BERT_OUTPUT_DIM, device=device)
        with torch.no_grad():
            encoded_traj, _, _, _, _ = model((traj, rand_traj, rand_nl))
            encoded_traj = encoded_traj.squeeze().detach().cpu().numpy()

        all_encoded_trajectories.append(encoded_traj)

    global GT_REWARD_MEAN
    global GT_REWARD_STD
    global SPEED_MEAN
    global SPEED_STD
    global HEIGHT_MEAN
    global HEIGHT_STD
    global DISTANCE_TO_BOTTLE_MEAN
    global DISTANCE_TO_BOTTLE_STD
    global DISTANCE_TO_CUBE_MEAN
    global DISTANCE_TO_CUBE_STD
    global ENCODED_TRAJECTORIES_MEAN
    global ENCODED_TRAJECTORIES_STD

    GT_REWARD_MEAN = np.mean(avg_gt_rewards)
    GT_REWARD_STD = np.std(avg_gt_rewards)
    SPEED_MEAN = np.mean(avg_speeds)
    SPEED_STD = np.std(avg_speeds)
    HEIGHT_MEAN = np.mean(avg_heights)
    HEIGHT_STD = np.std(avg_speeds)
    DISTANCE_TO_BOTTLE_MEAN = np.mean(avg_distance_to_bottles)
    DISTANCE_TO_BOTTLE_STD = np.std(avg_speeds)
    DISTANCE_TO_CUBE_MEAN = np.mean(avg_distance_to_cubes)
    DISTANCE_TO_CUBE_STD = np.std(avg_speeds)
    ENCODED_TRAJECTORIES_MEAN = np.mean(all_encoded_trajectories, axis=0)
    ENCODED_TRAJECTORIES_STD = np.std(all_encoded_trajectories, axis=0)


def make_gym_env(seed):
    # Create robosuite env
    env_kwargs = {
        "control_freq": 20,
        "env_name": "LiftModded",
        "hard_reset": False,
        "horizon": 500,
        "ignore_done": True,
        "reward_scale": 1.0,
        "robots": [
            "Jaco"
        ]
    }
    controller = "OSC_POSITION"
    controller_config = load_controller_config(default_controller=controller)
    env = suite.make(**env_kwargs,
                     has_renderer=False,
                     has_offscreen_renderer=True,
                     use_object_obs=True,
                     use_camera_obs=True,
                     reward_shaping=True,
                     controller_configs=controller_config)

    # Make sure we only pass in the proprio and object obs to the Gym env (no images)
    keys = ["object-state", "robot0_proprio-state"]

    # Make it a gym-compatible env
    gym_env = GymWrapper(env, keys=keys)
    obs_dim = gym_env.observation_space.low.size
    action_dim = gym_env.action_space.low.size

    np.random.seed(seed)
    gym_env.seed(seed)

    return gym_env


def load_model(model_path):
    # Load model
    device = torch.device(determine_default_torch_device(not torch.cuda.is_available()))
    print("device:", device)
    if device.type == 'cpu':  # device.type gets the actual string
        model = torch.load(model_path, map_location=torch.device('cpu'))
    else:
        model = torch.load(model_path)
    model.to(device)
    model.eval()
    return model, device


def run_aprel(seed, gym_env, model_path, human_user, traj_dir='', video_dir='', output_dir='', args=None):
    encoder_model, device = load_model(model_path)

    if args['normalize_feature_funcs']:
        trajs = np.concatenate((np.load(os.path.join(traj_dir, 'train/trajs.npy')), np.load(os.path.join(traj_dir, 'val/trajs.npy'))), axis=0)
        calc_and_set_global_vars(trajs, encoder_model, device)

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

    if traj_dir == '':
        trajectory_set = aprel.generate_trajectories_randomly(env, num_trajectories=10,
                                                              max_episode_length=500,
                                                              file_name="LiftModded", restore=True,
                                                              headless=False, seed=seed)
        val_trajectory_set = None
    else:
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

            traj = aprel.Trajectory(env, [(t[0:STATE_DIM], t[STATE_DIM:STATE_DIM+ACTION_DIM]) for t in train_traj], clip_path=clip_path)
            train_traj_set.append(traj)

        val_traj_set = []
        for i, val_traj in enumerate(val_trajs):
            # If available, load trajectories with camera observation.
            clip_path = None
            if videos_available:
                clip_path = os.path.join(video_dir, str(val_traj_video_ids[i]) + ".mp4")

            traj = aprel.Trajectory(env, [(t[0:STATE_DIM], t[STATE_DIM:STATE_DIM+ACTION_DIM]) for t in val_traj], clip_path=clip_path)
            val_traj_set.append(traj)

        trajectory_set = aprel.TrajectorySet(train_traj_set)
        val_trajectory_set = aprel.TrajectorySet(val_traj_set)

    features_dim = len(trajectory_set[0].features)

    query_optimizer = aprel.QueryOptimizerDiscreteTrajectorySet(trajectory_set)

    # Initialize the object for the true human
    if human_user:
        true_user = aprel.HumanUser(delay=0.5)
    else:
        # Create synthetic user with custom feature func
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

        # For the 'true' user reward, use a random length 5 vector
        # (user reward depends on gt_reward, speed, height, distance to bottle, distance to cube)
        true_features_dim = 5
        if args['query_type'] == 'nl_command':
            # NLCommandQuery requires the trajectory_set as one of the params
            true_params = {'weights': aprel.util_funs.get_random_normalized_vector(true_features_dim),
                           'beta': args['sim_user_beta'],
                           'feature_func': true_user_feature_func,
                           'trajectory_set': trajectory_set}
        else:
            true_params = {'weights': aprel.util_funs.get_random_normalized_vector(true_features_dim),
                           'beta': args['sim_user_beta'],
                           'feature_func': true_user_feature_func}
        print("True user parameters:", true_params['weights'])
        true_user = aprel.CustomFeatureUser(true_params)

    # Create the human response model and initialize the belief distribution
    if args['query_type'] == 'nl_command':
        # NLCommandQuery requires the trajectory_set as one of the params
        params = {'weights': aprel.util_funs.get_random_normalized_vector(features_dim),
                  'trajectory_set': trajectory_set}
        user_model = aprel.SoftmaxUser(params)
        params.pop('trajectory_set')
        belief = aprel.SamplingBasedBelief(user_model, [], params)
    else:
        params = {'weights': aprel.util_funs.get_random_normalized_vector(features_dim)}
        user_model = aprel.SoftmaxUser(params)
        belief = aprel.SamplingBasedBelief(user_model, [], params)
    print('Estimated user parameters: ' + str(belief.mean))

    # Initialize a dummy query so that the query optimizer will generate queries of the same kind
    if args['query_type'] == 'preference':
        query = aprel.PreferenceQuery(trajectory_set[:args['query_size']])
    elif args['query_type'] == 'nl_command':
        assert traj_dir != ''
        nl_comp_file = os.path.join(traj_dir, "train/unique_nlcomps_for_aprel.json")
        with open(nl_comp_file, 'rb') as f:
            nl_comps = json.load(f)

        try:
            nl_embedding_file = os.path.join(traj_dir, "train/unique_nlcomps_for_aprel.npy")
            nl_embeddings = np.load(nl_embedding_file)
        except FileNotFoundError:
            # Preprocess strings using BERT
            nl_embeddings = preprocess_strings('', 500, nl_comps)
        assert len(nl_comps) == len(nl_embeddings)

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
            rand_traj = torch.rand(1, 500, STATE_DIM+ACTION_DIM, device=device)
            with torch.no_grad():
                _, _, enc_str, _, _ = encoder_model((rand_traj, rand_traj, enc_str))
                enc_str = enc_str.squeeze().detach().cpu().numpy()

            return enc_str

        free_input = False
        print("free_input is", free_input)
        if free_input:
            query = aprel.NLCommandQuery(trajectory_set[:args['query_size']], lang_encoder_func)
        else:
            query = aprel.NLCommandQuery(trajectory_set[:args['query_size']], lang_encoder_func, nl_comps, nl_embeddings)
    elif args['query_type'] == 'weak_comparison':
        query = aprel.WeakComparisonQuery(trajectory_set[:args['query_size']])
    elif args['query_type'] == 'full_ranking':
        query = aprel.FullRankingQuery(trajectory_set[:args['query_size']])
    else:
        raise NotImplementedError('Unknown query type.')


    log_likelihoods = []
    val_log_likelihoods = []
    if not human_user:
        num_correct = 0
        num_incorrect = 0
        val_accuracies = []
    for query_no in range(args['num_iterations']):
        # Optimize the query
        print("Finding optimized query...")
        queries, objective_values = query_optimizer.optimize(args['acquisition'], belief,
                                                             query, batch_size=args['batch_size'],
                                                             optimization_method=args['optim_method'],
                                                             reduced_size=args['reduced_size_for_batches'],
                                                             gamma=args['dpp_gamma'],
                                                             distance=args['distance_metric_for_batches'])
        print('Objective Values: ' + str(objective_values))

        # Ask the query to the human
        responses = true_user.respond(queries)
        print("Response:", responses[0])

        # Update belief
        print("Updating belief via sampling...")
        initial_sampling_param = {"weights": [0 for _ in range(features_dim)]}
        if args['query_type'] == 'preference':
            belief.update(aprel.Preference(queries[0], responses[0]), initial_point=initial_sampling_param)
        elif args['query_type'] == 'nl_command':
            belief.update(aprel.NLCommand(queries[0], responses[0]), initial_point=initial_sampling_param)
        else:
            raise NotImplementedError('Unknown query type.')

        print('Estimated user parameters: ' + str(belief.mean))

        if not human_user:
            if args['query_type'] == 'preference':
                correct_true_user_response = np.argmax(true_user.response_logprobabilities(queries[0]))
            elif args['query_type'] == 'nl_command':
                correct_true_user_response_i = np.argmax(true_user.response_logprobabilities(queries[0]))
                correct_true_user_response = queries[0].response_set[correct_true_user_response_i]
            else:
                raise NotImplementedError('Unknown query type.')

            print("Correct response (based on true reward):", correct_true_user_response)
            if np.any(responses[0] != correct_true_user_response):
                print("Simulated human answered incorrectly!")
                num_incorrect += 1
            else:
                num_correct += 1

            # Question: why can't we use the already implemented `loglikelihood` function in SoftmaxUser? We're
            # essentially reimplementing that below.
            # Answer: We can, but the current user_model object was initialized with random weights, not the updated
            # weights belief.mean['weights'].
            if args['query_type'] == 'preference':
                # Old way, without reusing code (but maybe more efficient with garbage collection?)
                # ll = np.exp(np.dot(belief.mean['weights'], queries[0].slate[int(responses[0])].features))
                # ll /= np.exp(np.dot(belief.mean['weights'], queries[0].slate[int(responses[0])].features)) + np.exp(
                #     np.dot(belief.mean['weights'], queries[0].slate[1-int(responses[0])].features))
                # ll = np.log(ll)

                latest_params = {'weights': belief.mean['weights']}
                eval_user_model = aprel.SoftmaxUser(latest_params)
                ll = eval_user_model.loglikelihood(aprel.Preference(queries[0], responses[0]))
                print("log likelihood:", ll)
                log_likelihoods.append(ll)

            elif args['query_type'] == 'nl_command':
                # 1. Calculate log likelihood of response.
                latest_params = {'weights': belief.mean['weights'],
                                 'trajectory_set': trajectory_set}
                eval_user_model = aprel.SoftmaxUser(latest_params)
                ll = eval_user_model.loglikelihood(aprel.NLCommand(queries[0], responses[0]))
                print("log likelihood:", ll)
                log_likelihoods.append(ll)

                # 2. Find trajectory with highest return under the learned reward, and calculate the true reward.
            else:
                print('log likelihood calculation not supported for this query type yet.')

            if output_dir != '':
                np.save(os.path.join(output_dir, 'weights.npy'), belief.mean['weights'])
                np.save(os.path.join(output_dir, 'log_likelihoods.npy'), log_likelihoods)

            # Compute log likelihood on the set of val trajectories.
            # if args['query_type'] in ['preference', 'nl_command']:
            if val_trajectory_set is not None:
                val_lls = []
                val_num_correct = 0
                val_num_incorrect = 0
                if args['query_type'] == 'preference':
                    for i in range(val_trajectory_set.size):
                        for j in range(i+1, val_trajectory_set.size):
                            # Log likelihood under learned reward
                            val_query = aprel.PreferenceQuery([val_trajectory_set[i], val_trajectory_set[j]])
                            val_response = true_user.respond(val_query)
                            ll = eval_user_model.loglikelihood(aprel.Preference(val_query, val_response[0]))
                            val_lls.append(ll)

                            # Simulated user accuracy
                            correct_true_user_response = np.argmax(true_user.response_logprobabilities(val_query))

                            if val_response[0] != correct_true_user_response:
                                val_num_incorrect += 1
                            else:
                                val_num_correct += 1
                elif args['query_type'] == 'nl_command':
                    for i in range(val_trajectory_set.size):
                        val_query = queries[0].copy()
                        val_query.slate = [val_trajectory_set[i]]
                        val_response = true_user.respond(val_query)
                        ll = eval_user_model.loglikelihood(aprel.NLCommand(val_query, val_response[0]))
                        val_lls.append(ll)

                        correct_true_user_response_i = np.argmax(true_user.response_logprobabilities(val_query))
                        correct_true_user_response = val_query.response_set[correct_true_user_response_i]

                        if np.any(val_response[0] != correct_true_user_response):
                            val_num_incorrect += 1
                        else:
                            val_num_correct += 1
                else:
                    print('log likelihood calculation not supported for this query type yet.')

                print("validation log likelihood:", np.mean(val_lls))
                val_log_likelihoods.append(np.mean(val_lls))
                val_accuracy = val_num_correct / (val_num_correct + val_num_incorrect)
                val_accuracies.append(val_accuracy)
                if output_dir != '':
                    np.save(os.path.join(output_dir, 'val_log_likelihoods.npy'), val_log_likelihoods)
                    np.save(os.path.join(output_dir, 'simuser_val_accuracies.npy'), val_accuracies)
    if not human_user:
        train_accuracy = num_correct / (num_correct + num_incorrect)
        print("Accuracy of simulated user during active learning:", train_accuracy)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('--seed', type=int, default=0, help='')
    parser.add_argument('--model-path', type=str, default='', help='')
    parser.add_argument('--traj-dir', type=str, default='', help='')
    parser.add_argument('--video-dir', type=str, default='', help='')
    parser.add_argument('--human-user', action="store_true", help='')
    parser.add_argument('--sim-user-beta', type=float, default=1.0, help='')
    parser.add_argument('--output-dir', type=str, default='', help='')
    parser.add_argument('--query_type', type=str, default='preference',
                        help='Type of the queries that will be actively asked to the user. Options: preference, nl_command, weak_comparison, full_ranking.')
    parser.add_argument('--query_size', type=int, default=2,
                        help='Number of trajectories in each query.')
    parser.add_argument('--num_iterations', type=int, default=10,
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
    parser.add_argument('--normalize_feature_funcs', action="store_true", help='')


    args = parser.parse_args()

    seed = args.seed
    model_path = args.model_path
    traj_dir = args.traj_dir
    video_dir = args.video_dir
    human_user = args.human_user
    output_dir = args.output_dir
    args = vars(args)
    args['distance_metric_for_batches'] = aprel.default_query_distance # all relevant methods default to default_query_distance

    gym_env = make_gym_env(seed)

    run_aprel(seed, gym_env, model_path, human_user, traj_dir, video_dir, output_dir, args)

