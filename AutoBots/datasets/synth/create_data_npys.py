import argparse
import math
import os
import pickle

import numpy as np
from tqdm import tqdm

'''
Preprocess Synth-v1 into the format used by autobots.
'''


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


def prepare_data(raw_path, out_path, max_number_of_agents):
    with open(raw_path, "rb") as f:
        dataset = pickle.load(f)
    data = np.zeros((len(dataset["scenes"]), 20, max_number_of_agents, 2))
    largest_number_of_agents = 0
    for scene_idx, scene in enumerate(tqdm(dataset["scenes"])):
        trajectories = scene["trajectories"].transpose((1, 0, 2))
        assert len(trajectories) == 20
        if trajectories.shape[1] > largest_number_of_agents:
            largest_number_of_agents = trajectories.shape[1]
        trajectories = drop_distant(trajectories, max_num_peds=max_number_of_agents)
        trajectories, rotation, center = center_scene(trajectories)
        if trajectories.shape[1] < max_number_of_agents:
            tmp_trajectories = np.zeros((20, max_number_of_agents, 2))
            tmp_trajectories[:, :, :] = np.nan
            tmp_trajectories[:, :trajectories.shape[1], :] = trajectories
            trajectories = tmp_trajectories.copy()
        data[scene_idx] = trajectories
    np.save(out_path, data)
    print(f"Saved to {os.path.abspath(out_path)}")
    print(f"Largest number of agents in any scene: {largest_number_of_agents}")
    print(f"Maximum number of agents taken into account: {max_number_of_agents}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Synth-v1 Dataset Preprocessor.")
    parser.add_argument("--raw-dataset-path", type=str, required=True, help="Raw Synth-v1 pickle path.")
    parser.add_argument("--output-npy-path", type=str, required=True, help="Path to .npy to be outputted.")
    parser.add_argument("--max-number-of-agents", type=int, default=12, help="The maximum number of agents per scene")
    args = parser.parse_args()
    prepare_data(raw_path=args.raw_dataset_path, out_path=args.output_npy_path,
                 max_number_of_agents=args.max_number_of_agents)
    print("Done.")
