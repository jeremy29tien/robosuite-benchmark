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
from nl_traj_feature_learning.learn_features import BERT_OUTPUT_DIM

from gpu_utils import determine_default_torch_device
import argparse


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


def run_aprel(seed, gym_env, model_path, human_user, traj_file_path):
    encoder_model, device = load_model(model_path)

    def feature_func(traj):
        """Returns the features of the given trajectory, i.e. \Phi(traj).

        Args:
            traj: List of state-action tuples, e.g. [(state0, action0), (state1, action1), ...]

        Returns:
            features: a numpy vector corresponding the features of the trajectory
        """
        # print("type(traj[0][0]):", type(traj[0][0]))
        print("len(traj):", len(traj))
        traj = np.asarray([np.concatenate((t[0], t[1]), axis=0) for t in traj if t[1] is not None and t[0] is not None])
        traj = torch.unsqueeze(torch.as_tensor(traj, dtype=torch.float32, device=device), 0)
        rand_traj = torch.rand(traj.shape)
        rand_nl = torch.rand(1, BERT_OUTPUT_DIM)
        print("traj tensor:", traj.shape)
        with torch.no_grad():
            encoded_traj, _, _, _, _ = encoder_model((traj, rand_traj, rand_nl))
            encoded_traj = encoded_traj.squeeze().detach().cpu().numpy()

        # TODO: Could add a line that normalizes each feature in the embedding.
        #  Not sure whether there's a nice clean way of computing the mean and
        #  standard deviation of each feature though.
        return encoded_traj

    env = aprel.Environment(gym_env, feature_func)

    trajectory_set = aprel.generate_trajectories_randomly(env, num_trajectories=10,
                                                          max_episode_length=500,
                                                          file_name="LiftModded", restore=True,
                                                          headless=False, seed=seed)

    # Take trajectories from our training/val data.
    # trajectory_set = np.load(traj_file_path)

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
            features[0] = gt_reward(traj)
            features[1] = speed(traj)
            features[2] = height(traj)
            features[3] = distance_to_bottle(traj)
            features[4] = distance_to_cube(traj)

            return features

        # For the 'true' user reward, use a random length 5 vector
        # (user reward depends on gt_reward, speed, height, distance to bottle, distance to cube)
        true_features_dim = 5
        true_params = {'weights': aprel.util_funs.get_random_normalized_vector(true_features_dim),
                       'feature_func': true_user_feature_func}
        true_user = aprel.CustomFeatureUser(true_params)

    params = {'weights': aprel.util_funs.get_random_normalized_vector(features_dim)}
    user_model = aprel.SoftmaxUser(params)
    belief = aprel.SamplingBasedBelief(user_model, [], params)
    print('Estimated user parameters: ' + str(belief.mean))

    query = aprel.PreferenceQuery(trajectory_set[:2])

    for query_no in range(10):
        queries, objective_values = query_optimizer.optimize('mutual_information', belief, query)
        print('Objective Value: ' + str(objective_values[0]))

        responses = true_user.respond(queries[0])

        # Erdem's fix:
        # belief.update(aprel.Preference(queries[0], responses[0]))
        initial_sampling_param = {"weights": [0 for _ in range(features_dim)]}
        belief.update(aprel.Preference(queries[0], responses[0]), initial_point=initial_sampling_param)

        print('Estimated user parameters: ' + str(belief.mean))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('--seed', type=int, default=0, help='')
    parser.add_argument('--model-path', type=str, default='', help='')
    parser.add_argument('--traj-file-path', type=str, default='', help='')
    parser.add_argument('--human-user', action="store_true", help='')
    # parser.add_argument('--val-split', type=float, default=0.1, help='')

    args = parser.parse_args()

    seed = args.seed
    model_path = args.model_path
    traj_file_path = args.traj_file_path
    human_user = args.human_user

    gym_env = make_gym_env(seed)

    run_aprel(seed, gym_env, model_path, human_user, traj_file_path)

