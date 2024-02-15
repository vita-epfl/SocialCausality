import os

import numpy as np
from torch.utils.data import Dataset
import torch
import pickle
import math


def drop_distant(xy, max_num_peds=5):
    """
    Only Keep the max_num_peds closest pedestrians
    """
    distance_2 = np.sum(np.square(xy - xy[:, 0:1]), axis=2)
    smallest_dist_to_ego = np.nanmin(distance_2, axis=0)
    return xy[:, np.argsort(smallest_dist_to_ego)[:(max_num_peds)]]


def shift(xy, center):
    xy = xy - center[np.newaxis, np.newaxis, :]
    return xy


def theta_rotation(xy, theta):
    ct = math.cos(theta)
    st = math.sin(theta)

    r = np.array([[ct, st], [-st, ct]])
    return np.einsum('ptc,ci->pti', xy, r)


def center_scene(xy, obs_length=8, ped_id=0):
    ## Center
    center = xy[obs_length - 1, ped_id]  ## Last Observation
    xy = shift(xy, center)

    ## Rotate
    last_obs = xy[obs_length - 1, ped_id]
    second_last_obs = xy[obs_length - 2, ped_id]
    diff = np.array([last_obs[0] - second_last_obs[0], last_obs[1] - second_last_obs[1]])
    thet = np.arctan2(diff[1], diff[0])
    rotation = -thet + np.pi / 2
    xy = theta_rotation(xy, rotation)
    return xy, rotation, center


class SynthV1CausalDataset(Dataset):
    def __init__(self, dset_path, split="train", size=-1):
        # TODO: Note that the number of agents is hardcoded to match the
        #       standard Synth-v1. This parameter as well as the dataset
        #       preprocessing parameter `--max-number-of-agents` must be
        #       tweaked if a different dataset is used, e.g., one of the
        #       out-of-distribution datasets related to Synth-v1.
        self.num_others = 38

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
        else:
            self.size = min(self.size, len(os.listdir(self.dataset_path)))

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

    def _do_preprocess(self, raw_trajectories, max_number_of_agents=39):
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

        with open(os.path.join(self.dataset_path, "scene_"+str(idx)+".pkl"), "rb") as f:
            scene = pickle.load(f)

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
        # directly causality labels
        ego_causality_labels = scene["causality_labels"][0, :, 1:]
        directly_causal = ego_causality_labels.any(0)

        return f_cf_scenes, cf_causal_effects, directly_causal

    def __len__(self):
        return self.size


def my_collate_fn(batch):
    scenes, causal_effects, directly_causals, data_split = [], [], [], [0]
    for f_cf_scenes, cf_causal_effects, directly_causal in batch:
        scenes = scenes + f_cf_scenes
        causal_effects.append(cf_causal_effects)
        directly_causals.append(directly_causal)
        data_split.append(data_split[-1] + len(f_cf_scenes))
    return torch.utils.data.dataloader.default_collate(scenes), causal_effects, directly_causals, data_split