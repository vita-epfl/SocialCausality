import os

import numpy as np
from torch.utils.data import Dataset
import torch
import pickle
from datasets.synth.create_data_npys import drop_distant
from datasets.synth.create_data_npys import center_scene
from datasets.synth.create_data_npys import shift
from datasets.synth.create_data_npys import theta_rotation

class SynthV1Dataset(Dataset):
    def __init__(self, dset_path, filename, size=-1):
        # TODO: Note that the number of agents is hardcoded to match the
        #       standard Synth-v1. This parameter as well as the dataset
        #       preprocessing parameter `--max-number-of-agents` must be
        #       tweaked if a different dataset is used, e.g., one of the
        #       out-of-distribution datasets related to Synth-v1. 
        self.num_others = 11

        self.pred_horizon = 12
        self.num_agent_types = 1  # code assuming only one type of agent (pedestrians).
        self.in_seq_len = 8
        self.predict_yaw = False
        self.map_attr = 0  # dummy
        self.k_attr = 2
        dataset_path = os.path.join(dset_path, filename)
        print(f"SynthV1Dataset: Loading dataset from {os.path.abspath(dataset_path)}")
        self.agents_dataset = np.load(dataset_path)[:, :, :self.num_others + 1]
        if size != -1:
            self.agents_dataset = self.agents_dataset[:size]

    def unpack_datapoint(self, trajectories):
        assert len(trajectories) == self.in_seq_len + self.pred_horizon

        # Remove nan values and add mask column to state
        data_mask = np.ones((trajectories.shape[0], trajectories.shape[1], 3))
        data_mask[:, :, :2] = trajectories
        nan_indices = np.where(np.isnan(trajectories[:, :, 0]))
        data_mask[nan_indices] = [0, 0, 0]

        # Separate past and future.
        agents_in = data_mask[:self.in_seq_len]
        agents_out = data_mask[self.in_seq_len:]

        ego_in = agents_in[:, 0]
        ego_out = agents_out[:, 0]

        agent_types = np.ones((self.num_others + 1, self.num_agent_types))
        roads = np.ones((1, 1))  # for dataloading to work with other datasets that have images.

        return ego_in, ego_out, agents_in[:, 1:], agents_out[:, 1:], roads, agent_types

    def __getitem__(self, idx: int):
        trajectories = self.agents_dataset[idx]
        return self.unpack_datapoint(trajectories)

    def __len__(self):
        return len(self.agents_dataset)


class SynthV1CausalDataset(Dataset):
    def __init__(self, dset_path, split="train", size=-1):
        # TODO: Note that the number of agents is hardcoded to match the
        #       standard Synth-v1. This parameter as well as the dataset
        #       preprocessing parameter `--max-number-of-agents` must be
        #       tweaked if a different dataset is used, e.g., one of the
        #       out-of-distribution datasets related to Synth-v1.
        self.num_others = 11

        self.pred_horizon = 12
        self.num_agent_types = 1  # code assuming only one type of agent (pedestrians).
        self.in_seq_len = 8
        self.predict_yaw = False
        self.map_attr = 0  # dummy
        self.k_attr = 2

        self.dataset_path = os.path.join(dset_path, split)
        print(f"SynthV1CausalDataset: Loading dataset from {os.path.abspath(self.dataset_path)}")
        self.size = size
        if size == -1:
            self.size = len(os.listdir(self.dataset_path))
        # with open(dataset_path, "rb") as f:
        #     self.raw_dataset = pickle.load(f)["scenes"]
        # if size != -1:
        #     self.raw_dataset = self.raw_dataset[:size]
    def unpack_datapoint(self, trajectories):
        assert len(trajectories) == self.in_seq_len + self.pred_horizon

        # Remove nan values and add mask column to state
        data_mask = np.ones((trajectories.shape[0], trajectories.shape[1], 3))
        data_mask[:, :, :2] = trajectories
        nan_indices = np.where(np.isnan(trajectories[:, :, 0]))
        data_mask[nan_indices] = [0, 0, 0]

        # Separate past and future.
        agents_in = data_mask[:self.in_seq_len]
        agents_out = data_mask[self.in_seq_len:]

        ego_in = agents_in[:, 0]
        ego_out = agents_out[:, 0]

        agent_types = np.ones((self.num_others + 1, self.num_agent_types))
        roads = np.ones((1, 1))  # for dataloading to work with other datasets that have images.

        return ego_in, ego_out, agents_in[:, 1:], agents_out[:, 1:], roads, agent_types

    def _do_preprocess(self, raw_trajectories, max_number_of_agents=12):
        # Preprocess trajectories (center, rotate)
        assert len(raw_trajectories) == 20
        trajectories = drop_distant(raw_trajectories, max_num_peds=max_number_of_agents)
        trajectories, rotation, center = center_scene(trajectories)
        if trajectories.shape[1] < max_number_of_agents:
            tmp_trajectories = np.zeros((20, max_number_of_agents, 2))
            tmp_trajectories[:, :, :] = np.nan
            tmp_trajectories[:, :trajectories.shape[1], :] = trajectories
            trajectories = tmp_trajectories.copy()
        return self.unpack_datapoint(trajectories)

    def __getitem__(self, idx: int):
        f_cf_scenes = []
        cf_causal_effects = []

        # scene = self.raw_dataset[idx]
        with open(os.path.join(self.dataset_path, "scene_"+str(idx)+".pkl"), "rb") as f:
            scene = pickle.load(f)
        # self.raw_dataset = pickle.load(f)["scenes"]

        num_agents = len(scene["trajectories"])

        f_traj = scene["trajectories"].transpose((1, 0, 2))  # (time, agent, state)
        f_cf_scenes.append(self._do_preprocess(f_traj))

        for i in range(1, num_agents):
            cf_traj = scene["remove_agent_i_trajectories"][i].transpose((1, 0, 2))

            cf_gt = scene["remove_agent_i_trajectories"][i, 0, -self.pred_horizon:, :]
            f_gt = scene["trajectories"][0, -self.pred_horizon:, :]
            causal_effect = (np.sqrt(((cf_gt - f_gt) ** 2).sum(1))).mean()

            f_cf_scenes.append(self._do_preprocess(cf_traj))
            cf_causal_effects.append(causal_effect)

        return f_cf_scenes, cf_causal_effects

    def __len__(self):
        return self.size


def my_collate_fn(batch):
    scenes, causal_effects, data_split = [], [], [0]
    for f_cf_scenes, cf_causal_effects in batch:
        scenes = scenes + f_cf_scenes
        causal_effects.append(cf_causal_effects)
        data_split.append(data_split[-1] + len(f_cf_scenes))
    return torch.utils.data.dataloader.default_collate(scenes), causal_effects, data_split