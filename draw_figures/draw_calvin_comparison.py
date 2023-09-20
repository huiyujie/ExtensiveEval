import os
import sys
sys.path.append("..")
import matplotlib as mpl
from draw_figures.config import set_size
from draw_figures.draw_function import linestyle_tuple
from draw_figures.read_results import get_results
from matplotlib import cm, pyplot as plt
from draw_figures.config import config
import numpy as np

if __name__ == "__main__":
    mpl.rcParams.update(config)
    plt.rcParams['axes.edgecolor'] = '#333F4B'
    plt.rcParams['axes.linewidth'] = 0.8

    WIDTH = 345
    markers = ['o', 'v', '^', '<', '>', '1', '2', '3', '4']
    datas = []
    labels = []
    original_calvin = get_results(f"../results/ML/random/calvin_original.csv")
    aria_calvin = get_results(f"../results/ML/random/calvin_aria.csv")

    datas.append(original_calvin)
    labels.append("original calvin")

    datas.append(aria_calvin)
    labels.append("aria calvin")

    dim = set_size(WIDTH)
    width_ratio = 1.0
    fig, axe = plt.subplots(1, 1, figsize=(dim[0] * width_ratio, dim[1]))
    colmap = cm.get_cmap("Set1")
    colors =[colmap.colors[i] for i in range(9)]
    index = [f"{i:.2f}" for i in np.arange(0.05, 1, 0.05)]
    axe.set_ylim([0, 1])

    for i, r in enumerate(datas):
        error = []
        error.append(r["p50"] - r["p20"])
        error.append(r["p80"] - r["p50"])
        axe.errorbar(index, r["p50"], error, color=colors[i], marker=markers[i],
                     linestyle=linestyle_tuple[i][1], ms=1, mew=2, label=labels[i], linewidth=1.0)
    axe.set_xticklabels(index, rotation=30)
    axe.set_xlabel("Sampling Rate")
    axe.set_ylabel("P50 of R2 Scores +/- to P80 and P20")
    axe.legend(loc='upper left', bbox_to_anchor=(0., 1.1, 0, .102), ncol=2, borderaxespad=0.)

    print(f"save figure calvin comparison to figures/calvin_cmp.pdf")
    plt.savefig("figures/calvin_cmp.pdf", format='pdf', bbox_inches='tight')