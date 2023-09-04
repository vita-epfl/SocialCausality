from tensorflow.python.summary.summary_iterator import summary_iterator
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import pickle
import matplotlib.ticker as mtick


def plot_Autobot_datasize_effect_curve(show_percentage=False):
    datasizes = [1000, 2000, 5000, 10000, 20000, 50000, 100000]

    if os.path.isfile("datasize_effect_curve_datas.pkl"):
        with open("datasize_effect_curve_datas.pkl", "rb") as f:
            evaluation_metrics = pickle.load(f)
    else:
        evaluation_metrics = {"normal": {"ADE": [], "ARS": [], "HNC": []}}
        for exp in os.listdir("/scratch/izar/arahimi/autobots_causality/results/synth/"):
            if "_v2_ft_opt_" in exp and "contrastive_" in exp:
                weight = int(exp.split(':')[-1].replace("_v2_ft_opt_s1", ""))
                evaluation_metrics[f"contrastive_{weight}"] = {"ADE": [], "ARS": [], "HNC": []}
            elif "_v2_ft_opt_" in exp and "consistency_" in exp:
                weight = int(exp.split(':')[-1].replace("_v2_ft_opt_s1", ""))
                evaluation_metrics[f"consistency_{weight}"] = {"ADE": [], "ARS": [], "HNC": []}
        print("datasize", "saving", "ade", "ars", "hnc")
        for datasize in datasizes:
            for saving in evaluation_metrics.keys():
                                     # "/home/arahimi/AutoBots/results/synth_causal/Autobot_ego_C1_H128_E2_D2_TXH384_NH16_EW40_KLW20_NormLoss_synth-contrast-scenes:{}-cw:20000-datasize-check_s1/tb_files",
                                     # "/home/arahimi/AutoBots/results/synth_causal/Autobot_ego_C1_H128_E2_D2_TXH384_NH16_EW40_KLW20_NormLoss_synth-consis-scenes:{}-cw:10-datasize-check_s1/tb_files"]):
                if saving == "normal":
                    log_directory = "/scratch/izar/arahimi/autobots_causality/results/synth/Autobot_ego_NScene:{}_regType:None_v2_ft_opt_s1/tb_files".format(datasize)
                else:
                    reg, weight = saving.split('_')
                    log_directory = "/scratch/izar/arahimi/autobots_causality/results/synth/Autobot_ego_NScene:{}_regType:{}_CW:{}_v2_ft_opt_s1/tb_files".format(datasize, reg, weight)
                # log_directory = path.format(datasize)
                log_file = os.path.join(log_directory, os.listdir(log_directory)[0])
                ade, ars, hnc = np.inf, np.inf, np.inf
                try:
                    for summary in summary_iterator(log_file):
                        try:
                            if len(summary.summary.value) == 0:  # not a record in our dataset
                                continue
                            if summary.summary.value[0].tag == "metrics/Val minADE_1":
                                ade = min(ade, summary.summary.value[0].simple_value)
                            if summary.summary.value[0].tag == "metrics/Val ARS":
                                ars = min(ars, summary.summary.value[0].simple_value)
                            if summary.summary.value[0].tag == "metrics/Val HNC":
                                hnc = min(hnc, summary.summary.value[0].simple_value)
                        except:
                            breakpoint()
                    evaluation_metrics[saving]["ADE"].append(ade)
                    evaluation_metrics[saving]["ARS"].append(ars)
                    evaluation_metrics[saving]["HNC"].append(hnc)
                    print(datasize, saving, ade, ars, hnc)
                except:
                    continue
        with open("datasize_effect_curve_datas.pkl", "wb") as f:
            pickle.dump(evaluation_metrics, f)

    if show_percentage:
        datasizes = np.array(datasizes) / 100000 * 100
    for metric in ["ARS", "HNC", "ADE"]:
        height = 6
        fig, ax = plt.subplots(figsize=(height, height))
        # for data, color, label in zip([evaluation_metrics[saving][metric] for saving in ["normal", "contrast", "consis"]],
        #                               ["GoldenRod", sns.xkcd_rgb["slime green"], "orangered"],
        #                               ["Baseline", "Contrastive", "Consistency"]):
        for key, value in evaluation_metrics.items():
            data = value[metric]
            print(key, datasizes, data)
            if len(data) > 0 and ("consistency" in key or "normal" in key):
                sns.lineplot(
                    x=datasizes,
                    y=data,
                    # c=color,
                    marker="o",
                    linestyle="-",
                    linewidth=2,
                    alpha=1,
                    ax=ax,
                    label=key
                )

        if show_percentage:
            ax.xaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))

        plt.xlabel("Used data")
        # plt.xscale("log")
        plt.ylabel(metric)

        # plt.legend(loc="upper left")
        plt.tight_layout()

        # os.makedirs("Autobot", exist_ok=True)
        plt.savefig(f'data_curve_{metric}.png')
        plt.savefig(f'data_curve_{metric}.pdf')
        # plt.show()

if __name__ == '__main__':
    plot_Autobot_datasize_effect_curve()