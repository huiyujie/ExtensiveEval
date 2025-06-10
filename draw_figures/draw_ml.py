import os
import sys
sys.path.append("..")
import util_func
import matplotlib as mpl
import matplotlib.pyplot as plt

from draw_figures.config import config
from draw_figures.draw_function import draw_one_sys, draw_combined_anova_ML, draw_multiple_systes,  draw_multiple_systes_xgb

if __name__ == "__main__":
    mpl.rcParams.update(config)
    plt.rcParams['axes.edgecolor'] = '#333F4B'
    plt.rcParams['axes.linewidth'] = 0.8


    # for sys_conf in util_func.GOOD_SYS:
    #     draw_combined_anova_ML(sys_conf, "good", "random", save=True)
    # for sys_conf in util_func.BAD_SYS:
    #     draw_combined_anova_ML(sys_conf, "bad", "random", save=True)

    # draw_multiple_systes([("mysql", "tpcc"),
    #         ("aria","tpcc","Aria"),
    #         ("star","ycsb"),
    #         ("herd","ycsb")], "good", "random", save=True)

    # draw_multiple_systes([
    #        ("calvin","tpcc","Calvin"),
    #        ("drtm","tpcc"),
    #        ("star","tpcc"),
    #        ("silo","ycsb")], "bad", "random", save=True)
    
    draw_multiple_systes_xgb([("mysql", "tpcc"),
            ("aria","tpcc","Aria"),
            ("star","ycsb"),
            ("herd","ycsb")], "good", "random", save=True)

    draw_multiple_systes_xgb([
           ("calvin","tpcc","Calvin"),
           ("drtm","tpcc"),
           ("star","tpcc"),
           ("silo","ycsb")], "bad", "random", save=True)

