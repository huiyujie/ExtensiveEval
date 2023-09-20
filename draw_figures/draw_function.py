from matplotlib import cm, pyplot as plt

from draw_figures.config import set_size
import pandas as pd
import numpy as np

from draw_figures.helper import pick_best_stratified
from draw_figures.read_results import get_results

WIDTH = 345
linestyle_tuple = [
     ('solid',        (0, ())),
     ('dotted',                (0, (1, 1))),
     ('long dash with offset', (5, (10, 3))),
     ('dashdotted',            (0, (3, 5, 1, 5))),
     ('densely dashdotted',    (0, (3, 1, 1, 1))),

     ('dashdotdotted',         (0, (3, 5, 1, 5, 1, 5))),
     ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
     ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))]

def draw_error_multiplelines(combs, read_dir, system_type, save_file, save=False):
    markers = ['o' ,'v' ,'^' ,'<' ,'>', '1', '2', '3', '4']
    datas = []
    labels = []
    for comb in combs:
        filename = comb[0]+"-"+comb[1]+"-"+comb[2] if len(comb)==3 else comb[0]+"-"+comb[1]
        path = f"{read_dir}/{filename}.csv"
        data = get_results(path)
        datas.append(data)
        labels.append(filename)
    dim = set_size(WIDTH)
    width_ratio = 1.0
    fig, axe = plt.subplots(1, 1, figsize=(dim[0] * width_ratio, dim[1]))
    colmap = cm.get_cmap("Set1")
    colors =[colmap.colors[i] for i in range(9)]
    index = [f"{i:.2f}" for i in np.arange(0.05, 1, 0.05)]
    if system_type == "good":
        axe.set_ylim([0.8, 1.05])
    elif system_type == "bad":
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
    axe.legend(loc='upper left', bbox_to_anchor=(0., 1.3, 0, .102), ncol=2, borderaxespad=0.)


    if save:
        plt.savefig("figures/" + save_file + ".pdf", format='pdf', bbox_inches='tight')


def draw_one_sys(comb, factor_names, system_type, save=False):
    markers = ['o' ,'v' ,'^' ,'<' ,'>', '1', '2', '3', '4']
    datas = []
    labels = []
    filename = comb[0] + "-" + comb[1] + "-" + comb[2] if len(comb) == 3 else comb[0] + "-" + comb[1]
    #random
    data = get_results(f"../results/ML/random/{filename}.csv")
    datas.append(data)
    labels.append("random")
    #balance
    data = get_results(f"../results/ML/balance/{filename}.csv")
    datas.append(data)
    labels.append("balance")
    #dist-aware
    data = get_results(f"../results/ML/dist-aware/{filename}.csv")
    datas.append(data)
    labels.append("dist-aware")
    #stratified
    factor = pick_best_stratified(filename, factor_names)
    data = get_results(f"../results/ML/stratified/{filename}-{factor}.csv")
    datas.append(data)
    labels.append("stratified")

    dim = set_size(WIDTH)
    width_ratio = 1.0
    fig, axe = plt.subplots(1, 1, figsize=(dim[0] * width_ratio, dim[1]))
    colmap = cm.get_cmap("Set1")
    colors =[colmap.colors[i] for i in range(9)]
    index = [f"{i:.2f}" for i in np.arange(0.05, 1, 0.05)]
    if system_type == "good":
        axe.set_ylim([0.8, 1.05])
    elif system_type == "bad":
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


    if save:
        print(f"save figure {filename} to figures/{filename}.pdf")
        plt.savefig("figures/" + filename + ".pdf", format='pdf', bbox_inches='tight')