import torch
from torch.utils.data import Dataset
import json
import numpy as np
import argparse
import os
from robosuite.synthetic_comparisons import generate_synthetic_comparisons_commands, generate_noisyaugmented_synthetic_comparisons_commands, calc_and_set_global_vars


def get_comparisons(traj_i, traj_j, traj_i_reward, traj_j_reward, noise_augmentation=0):
    out = []
    if noise_augmentation == 0:
        gt_reward_comps = generate_synthetic_comparisons_commands(traj_i, traj_j, traj_i_reward,
                                                                           traj_j_reward, 'gt_reward')
        speed_comps = generate_synthetic_comparisons_commands(traj_i, traj_j, traj_i_reward, traj_j_reward,
                                                                       'speed')
        height_comps = generate_synthetic_comparisons_commands(traj_i, traj_j, traj_i_reward, traj_j_reward,
                                                                        'height')
        distance_to_bottle_comps = generate_synthetic_comparisons_commands(traj_i, traj_j, traj_i_reward,
                                                                                    traj_j_reward,
                                                                                    'distance_to_bottle')
        distance_to_cube_comps = generate_synthetic_comparisons_commands(traj_i, traj_j, traj_i_reward,
                                                                                  traj_j_reward, 'distance_to_cube')
    else:
        gt_reward_comps = generate_noisyaugmented_synthetic_comparisons_commands(traj_i, traj_j, traj_i_reward,
                                                                           traj_j_reward, 'gt_reward', n_duplicates=noise_augmentation)
        speed_comps = generate_noisyaugmented_synthetic_comparisons_commands(traj_i, traj_j, traj_i_reward, traj_j_reward,
                                                                       'speed', n_duplicates=noise_augmentation)
        height_comps = generate_noisyaugmented_synthetic_comparisons_commands(traj_i, traj_j, traj_i_reward, traj_j_reward,
                                                                        'height', n_duplicates=noise_augmentation)
        distance_to_bottle_comps = generate_noisyaugmented_synthetic_comparisons_commands(traj_i, traj_j, traj_i_reward,
                                                                                    traj_j_reward,
                                                                                    'distance_to_bottle', n_duplicates=noise_augmentation)
        distance_to_cube_comps = generate_noisyaugmented_synthetic_comparisons_commands(traj_i, traj_j, traj_i_reward,
                                                                                  traj_j_reward, 'distance_to_cube', n_duplicates=noise_augmentation)

    for c in gt_reward_comps + speed_comps + height_comps + distance_to_bottle_comps + distance_to_cube_comps:
        out.append(c)
    return out


def generate_dataset(trajs, traj_rewards, noise_augmentation=0, id_mapping=False, all_pairs=True, dataset_size=0):
    dataset_traj_as = []
    dataset_traj_bs = []
    dataset_comps = []
    num_trajectories = len(trajs)

    # Prep work for noisy data augmentation
    if noise_augmentation:
        calc_and_set_global_vars(trajs, traj_rewards)

    if all_pairs:
        print("GENERATING USING ALL-PAIRS METHOD.")
        for i in range(0, num_trajectories):
            print("GENERATING COMPARISONS FOR i =", i)
            for j in range(i+1, num_trajectories):
                traj_i = trajs[i]
                traj_j = trajs[j]
                traj_i_reward = traj_rewards[i]
                traj_j_reward = traj_rewards[j]

                comps = get_comparisons(traj_i, traj_j, traj_i_reward, traj_j_reward, noise_augmentation=noise_augmentation)
                flipped_comps = get_comparisons(traj_j, traj_i, traj_j_reward, traj_i_reward, noise_augmentation=noise_augmentation)

                if id_mapping:  # With this option, we store the indexes of the `trajs` array rather than the actual trajectory
                    for c in comps:
                        dataset_traj_as.append(i)
                        dataset_traj_bs.append(j)
                        dataset_comps.append(c)
                    for fc in flipped_comps:
                        dataset_traj_as.append(j)
                        dataset_traj_bs.append(i)
                        dataset_comps.append(fc)
                else:
                    for c in comps:
                        dataset_traj_as.append(traj_i)
                        dataset_traj_bs.append(traj_j)
                        dataset_comps.append(c)
                    for fc in flipped_comps:
                        dataset_traj_as.append(traj_j)
                        dataset_traj_bs.append(traj_i)
                        dataset_comps.append(fc)

    else:
        print("GENERATING " + str(dataset_size) + " RANDOM COMPARISONS.")
        for n in range(dataset_size):
            print("GENERATING COMPARISONS FOR n =", n)
            i = 0
            j = 0
            while i == j:
                i = np.random.randint(num_trajectories)
                j = np.random.randint(num_trajectories)

            traj_i = trajs[i]
            traj_j = trajs[j]
            traj_i_reward = traj_rewards[i]
            traj_j_reward = traj_rewards[j]

            comps = get_comparisons(traj_i, traj_j, traj_i_reward, traj_j_reward, noise_augmentation=noise_augmentation)
            flipped_comps = get_comparisons(traj_j, traj_i, traj_j_reward, traj_i_reward, noise_augmentation=noise_augmentation)

            if id_mapping:  # With this option, we store the indexes of the `trajs` array rather than the actual trajectory
                for c in comps:
                    dataset_traj_as.append(i)
                    dataset_traj_bs.append(j)
                    dataset_comps.append(c)
                for fc in flipped_comps:
                    dataset_traj_as.append(j)
                    dataset_traj_bs.append(i)
                    dataset_comps.append(fc)
            else:
                for c in comps:
                    dataset_traj_as.append(traj_i)
                    dataset_traj_bs.append(traj_j)
                    dataset_comps.append(c)
                for fc in flipped_comps:
                    dataset_traj_as.append(traj_j)
                    dataset_traj_bs.append(traj_i)
                    dataset_comps.append(fc)

    return dataset_traj_as, dataset_traj_bs, dataset_comps


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('--policy-dir', type=str, default='', help='')
    parser.add_argument('--output-dir', type=str, default='', help='')
    parser.add_argument('--dataset-size', type=int, default=1000, help='')
    parser.add_argument('--noise-augmentation', type=int, default=0, help='')
    parser.add_argument('--id-mapping', action="store_true", help='')
    parser.add_argument('--all-pairs', action="store_true", help='')
    parser.add_argument('--trajs-per-policy', type=int, default=5, help='')
    parser.add_argument('--val-split', type=float, default=0.1, help='')
    parser.add_argument('--seed', type=int, default=0, help='')

    args = parser.parse_args()

    policy_dir = args.policy_dir
    output_dir = args.output_dir
    noise_augmentation = args.noise_augmentation
    id_mapping = args.id_mapping
    dataset_size = args.dataset_size
    all_pairs = args.all_pairs
    trajs_per_policy = args.trajs_per_policy
    val_split = args.val_split
    seed = args.seed

    np.random.seed(seed)

    print("GETTING TRAJECTORY ROLLOUTS...")
    trajectories = []
    trajectory_rewards = []
    for config in os.listdir(policy_dir):
        policy_path = os.path.join(policy_dir, config)
        if os.path.isdir(policy_path) and os.listdir(
                policy_path):  # Check that policy_path is a directory and that directory is not empty
            print(policy_path)
            observations = np.load(os.path.join(policy_path, "traj_observations.npy"))
            actions = np.load(os.path.join(policy_path, "traj_actions.npy"))
            rewards = np.load(os.path.join(policy_path, "traj_rewards.npy"))
            # observations has dimensions (n_trajs, n_timesteps, obs_dimension)
            trajs = np.concatenate((observations, actions), axis=-1)

            # Downsample
            trajs = trajs[0:trajs_per_policy]
            rewards = rewards[0:trajs_per_policy]

            # NOTE: We use extend rather than append because we don't want to add an
            # additional dimension across the policies.
            trajectories.extend(trajs)
            trajectory_rewards.extend(rewards)

    trajectories = np.asarray(trajectories)
    trajectory_rewards = np.asarray(trajectory_rewards)
    num_trajectories = trajectories.shape[0]

    # Shuffle
    p = np.random.permutation(num_trajectories)
    trajectories = trajectories[p]
    trajectory_rewards = trajectory_rewards[p]

    # Split
    split_i = int(np.ceil(val_split*num_trajectories))
    val_trajectories = trajectories[0:split_i]
    val_trajectory_rewards = trajectory_rewards[0:split_i]
    train_trajectories = trajectories[split_i:]
    train_trajectory_rewards = trajectory_rewards[split_i:]

    print("NUM_TRAJECTORIES:", num_trajectories)
    print("NUM TRAIN TRAJECTORIES:", len(train_trajectories))
    print("NUM VAL TRAJECTORIES:", len(val_trajectories))
    print("COMPILING DATASET:")

    train_traj_as, train_traj_bs, train_comps = generate_dataset(train_trajectories, train_trajectory_rewards,
                                                                 noise_augmentation=noise_augmentation, id_mapping=id_mapping, all_pairs=all_pairs, dataset_size=dataset_size)
    val_traj_as, val_traj_bs, val_comps = generate_dataset(val_trajectories, val_trajectory_rewards,
                                                           id_mapping=id_mapping, all_pairs=True)

    if id_mapping:
        np.save(os.path.join(output_dir, 'train/traj_a_indexes.npy'), train_traj_as)
        np.save(os.path.join(output_dir, 'train/traj_b_indexes.npy'), train_traj_bs)
        np.save(os.path.join(output_dir, 'train/trajs.npy'), train_trajectories)
        # TODO: save trajectory rewards too.
        with open(os.path.join(output_dir, 'train/nlcomps.json'), 'w') as f:
            json.dump(train_comps, f)
        np.save(os.path.join(output_dir, 'val/traj_a_indexes.npy'), val_traj_as)
        np.save(os.path.join(output_dir, 'val/traj_b_indexes.npy'), val_traj_bs)
        np.save(os.path.join(output_dir, 'val/trajs.npy'), val_trajectories)
        with open(os.path.join(output_dir, 'val/nlcomps.json'), 'w') as f:
            json.dump(val_comps, f)
    else:
        np.save(os.path.join(output_dir, 'train/traj_as.npy'), train_traj_as)
        np.save(os.path.join(output_dir, 'train/traj_bs.npy'), train_traj_bs)
        with open(os.path.join(output_dir, 'train/nlcomps.json'), 'w') as f:
            json.dump(train_comps, f)
        np.save(os.path.join(output_dir, 'val/traj_as.npy'), val_traj_as)
        np.save(os.path.join(output_dir, 'val/traj_bs.npy'), val_traj_bs)
        with open(os.path.join(output_dir, 'val/nlcomps.json'), 'w') as f:
            json.dump(val_comps, f)



