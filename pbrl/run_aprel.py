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


controller = "OSC_POSITION"
controller_config = load_controller_config(default_controller=controller)
env = suite.make(has_renderer=False, has_offscreen_renderer=False, use_object_obs=True, use_camera_obs=False,
                 reward_shaping=True, controller_configs=controller_config)

# Create gym-compatible env
gym_env = GymWrapper(env)

obs_dim = gym_env.observation_space.low.size
action_dim = gym_env.action_space.low.size

# env_name = 'MountainCarContinuous-v0'
# gym_env = gym.make(env_name)

np.random.seed(0)
gym_env.seed(0)


def load_model(model_path):
    # Load model
    device = torch.device(determine_default_torch_device(not torch.cuda.is_available()))
    print("device:", device)
    model = torch.load(model_path)
    model.to(device)
    model.eval()
    return model, device


def feature_func(traj):
    """Returns the features of the given MountainCar trajectory, i.e. \Phi(traj).

    Args:
        traj: List of state-action tuples, e.g. [(state0, action0), (state1, action1), ...]

    Returns:
        features: a numpy vector corresponding the features of the trajectory
    """
    # TODO: Using the trained model, encode the trajectory to get our 16-dimensional 'feature' encoding.

    states = np.array([pair[0] for pair in traj])
    actions = np.array([pair[1] for pair in traj[:-1]])
    min_pos, max_pos = states[:, 0].min(), states[:, 0].max()
    mean_speed = np.abs(states[:, 1]).mean()
    mean_vec = [-0.703, -0.344, 0.007]
    std_vec = [0.075, 0.074, 0.003]
    return (np.array([min_pos, max_pos, mean_speed]) - mean_vec) / std_vec


env = aprel.Environment(gym_env, feature_func)

trajectory_set = aprel.generate_trajectories_randomly(env, num_trajectories=10,
                                                      max_episode_length=300,
                                                      file_name=env_name, seed=0)
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
