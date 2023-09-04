import numpy as np
import os

for sulrm_file in ["slurm-1466792.out", "slurm-1466793.out", "slurm-1466794.out", "slurm-1466795.out"]:
    with open(sulrm_file, "r") as f:
        lines = f.readlines()
        synth_eval = False
        metrics = {"eth_ucy_ade":[], "eth_ucy_fde":[]}
        for line in lines:
            if "minADE_1" in line:
                if not synth_eval:
                    metrics["synth_minADE"] = round(float(line.split()[1]), 3)
                    metrics["synth_minFDE"] = round(float(line.split()[7]), 3)
                    metrics["synth_HNC"] = float(line.split()[-3])
                    metrics["synth_ARS"] = float(line.split()[-1])
                    synth_eval = True
                else:
                    metrics["eth_ucy_ade"].append(round(float(line.split()[1]), 3))
                    metrics["eth_ucy_fde"].append(round(float(line.split()[7]), 3))
        print(metrics["synth_minADE"], metrics["synth_minFDE"], metrics["synth_HNC"], metrics["synth_ARS"], end=" ")
        for i in range(5):
            print(str(metrics["eth_ucy_ade"][i]) + "/" + str(metrics["eth_ucy_fde"][i]), end=" ")
        print(str(round(np.mean(metrics["eth_ucy_ade"]), 3)) + "/" + str(round(np.mean(metrics["eth_ucy_fde"]), 3)))