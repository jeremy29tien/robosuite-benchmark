import torch
from torch.utils.data import Dataset
import json
import numpy as np


# nlcomp_file is a json file with the list of comparisons in NL.
# traj_a_file is a .npy or .npz file with the first trajectory and has a shape of (n_trajectories, n_timesteps, STATE_DIM+ACTION_DIM)
# traj_b_file is a .npy or .npz file with the second trajectory and has a shape of (n_trajectories, n_timesteps, STATE_DIM+ACTION_DIM)
class NLTrajComparisonDataset(Dataset):
    def __init__(self, nlcomp_file, traj_a_file, traj_b_file, preprocessed_nlcomps=False):
        if preprocessed_nlcomps:
            self.nlcomps = np.load(nlcomp_file)
        else:
            with open(nlcomp_file, 'rb') as f:
                self.nlcomps = json.load(f)
        # self.traj_as = np.load(traj_a_file)
        # NOTE: `mmap_mode` memory-maps the file, enabling us to read directly from disk
        # rather than loading the entire (huge) array into memory.
        self.traj_as = np.load(traj_a_file, mmap_mode='r')
        self.traj_bs = np.load(traj_b_file, mmap_mode='r')

    def __len__(self):
        return len(self.nlcomps)

    def __getitem__(self, idx):
        traj_a = self.traj_as[idx, :, :]
        traj_b = self.traj_bs[idx, :, :]
        nlcomp = self.nlcomps[idx]
        return traj_a, traj_b, nlcomp
