import numpy as np
import os
import pickle
from tqdm import tqdm


if __name__ == '__main__':
    
    data_path = '/work/vita/datasets/causal_synthetic_data/synth_v1.a.filtered.{}.pkl'
    # data_path = '/work/vita/datasets/causal_synthetic_data/synth_v1.a.{}.val.300.pkl'
    save_path = '/scratch/izar/luan/synth_v1_breaked/{}'
    for split in ["train", "val"]:
    # for dataset_name in ["filtered"]:
        os.makedirs(save_path.format(split), exist_ok=True)

        with open(data_path.format(split), "rb") as f:
            data = pickle.load(f)["scenes"]
        # num_agents, num_c_agents, num_nc_agents = [], [], []
        for i, scene in enumerate(tqdm(data)):
            with open(os.path.join(save_path.format(split), "scene_"+str(i)+".pkl"), "wb") as f:
                pickle.dump(scene, f)
            # num_agents.append(len(scene["trajectories"]))
            # num_c_agents.append((scene["remove_agent_i_ade"][1:, 0] >= 0.1).sum())
            # num_nc_agents.append((scene["remove_agent_i_ade"][1:, 0] <= 0.02).sum())
        print(split)
        # print("avg # (agents:{:.3f}, c agents:{:.3f}, nc_agents:{:.3f}".format(np.array(num_agents).mean(), np.array(num_c_agents).mean(), np.array(num_nc_agents).mean()))
