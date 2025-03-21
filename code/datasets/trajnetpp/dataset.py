import os
import numpy as np
import glob
from torch.utils.data import Dataset


class TrajNetPPDataset(Dataset):
    def __init__(self, dset_path, split_name="train", proportion=1.0):
        self.num_others = 38
        self.pred_horizon = 12
        self.num_agent_types = 1  # code assuming only one type of agent (pedestrians).
        self.in_seq_len = 8
        self.predict_yaw = False
        self.map_attr = 0  # dummy
        self.k_attr = 2

        dset_fnames = sorted(glob.glob(os.path.join(dset_path, split_name+"_*.npy")))
        
        agents_dataset = []
        for dset_fname in dset_fnames:
            agents_dataset.append(np.load(dset_fname))
        self.agents_dataset = np.concatenate(agents_dataset)[:, :, :self.num_others+1]
        # for low-data regimes
        if proportion != 1.0:
            # shuffle 
            dara_len = int(self.agents_dataset.shape[0] * proportion)
            self.agents_dataset = np.random.permutation(self.agents_dataset)
            self.agents_dataset = self.agents_dataset[:dara_len]

        del agents_dataset

    def __getitem__(self, idx: int):
        data = self.agents_dataset[idx]
        if data.shape[1] < self.num_others + 1:
                # Need to pad array to have shape 21xNx2
                temp_curr_scene = np.zeros((20, self.num_others+1, 2))
                temp_curr_scene[:, :, :] = np.nan
                temp_curr_scene[:, :data.shape[1], :] = data
                data = temp_curr_scene.copy()
        # Remove nan values and add mask column to state
        data_mask = np.ones((data.shape[0], data.shape[1], 3))
        data_mask[:, :, :2] = data
        nan_indices = np.where(np.isnan(data[:, :, 0]))
        data_mask[nan_indices] = [0, 0, 0]

        # Separate past and future.
        agents_in = data_mask[:self.in_seq_len]
        agents_out = data_mask[self.in_seq_len:]

        ego_in = agents_in[:, 0]
        ego_out = agents_out[:, 0]

        agent_types = np.ones((self.num_others + 1, self.num_agent_types))
        roads = np.ones((1, 1))  # for dataloading to work with other datasets that have images.

        return ego_in, ego_out, agents_in[:, 1:], roads
    

    def __len__(self):
        return len(self.agents_dataset)

