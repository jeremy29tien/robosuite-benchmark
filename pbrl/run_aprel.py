import aprel
import numpy as np
import gym

import robosuite as suite
from robosuite.wrappers import GymWrapper
from robosuite.controllers import load_controller_config, ALL_CONTROLLERS

import torch
from nl_traj_feature_learning.learn_features import NLTrajAutoencoder
from nl_traj_feature_learning.nl_traj_dataset import NLTrajComparisonDataset

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
                     has_offscreen_renderer=False,
                     use_object_obs=True,
                     use_camera_obs=False,
                     reward_shaping=True,
                     controller_configs=controller_config)

    # Make it a gym-compatible env
    gym_env = GymWrapper(env)
    obs_dim = gym_env.observation_space.low.size
    action_dim = gym_env.action_space.low.size

    np.random.seed(seed)
    gym_env.seed(seed)

    return gym_env


def load_model(model_path):
    # Load model
    device = torch.device(determine_default_torch_device(not torch.cuda.is_available()))
    print("device:", device)
    model = torch.load(model_path)
    model.to(device)
    model.eval()
    return model, device


def run_aprel(gym_env, model_path):
    encoder_model, device = load_model(model_path)

    def feature_func(traj):
        """Returns the features of the given trajectory, i.e. \Phi(traj).

        Args:
            traj: List of state-action tuples, e.g. [(state0, action0), (state1, action1), ...]

        Returns:
            features: a numpy vector corresponding the features of the trajectory
        """
        traj = [t[0] + t[1] for t in traj]
        traj = torch.unsqueeze(torch.as_tensor(traj, dtype=torch.float32, device=device), 0)
        print("traj tensor:", traj.shape)
        with torch.no_grad():
            encoded_traj, _, _, _, _ = encoder_model((traj, traj, ""))
            encoded_traj = encoded_traj.squeeze().detach().cpu().numpy()

        # TODO: Could add a line that normalizes each feature in the embedding.
        #  Not sure whether there's a nice clean way of computing the mean and
        #  standard deviation of each feature though.
        return encoded_traj

    env = aprel.Environment(gym_env, feature_func)

    # trajectory_set = aprel.generate_trajectories_randomly(env, num_trajectories=10,
    #                                                       max_episode_length=300,
    #                                                       file_name=env_name, seed=0)

    # TODO: take trajectories from our training/val data.
    trajectory_set = None

    # TODO: modify features_dim to reflect the feature dimension (16).
    #  This is later used in `aprel.util_funs.get_random_normalized_vector(features_dim)`
    features_dim = len(trajectory_set[0].features)

    query_optimizer = aprel.QueryOptimizerDiscreteTrajectorySet(trajectory_set)

    true_user = aprel.HumanUser(delay=0.5)

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
    # parser.add_argument('--id-mapping', action="store_true", help='')
    # parser.add_argument('--val-split', type=float, default=0.1, help='')

    args = parser.parse_args()

    seed = args.seed
    model_path = args.model_path

    gym_env = make_gym_env(seed)

    run_aprel(gym_env, model_path)

