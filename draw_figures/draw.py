import os
import sys
sys.path.append("..")
import util_func
import matplotlib as mpl
import matplotlib.pyplot as plt

from draw_figures.config import config
from draw_figures.draw_function import draw_one_sys, draw_combined_anova_ML

if __name__ == "__main__":
    mpl.rcParams.update(config)
    plt.rcParams['axes.edgecolor'] = '#333F4B'
    plt.rcParams['axes.linewidth'] = 0.8


    for sys_conf in util_func.GOOD_SYS:
        # draw_one_sys(sys_conf, util_func.get_factor_names(sys_conf), "good", save=True)
        draw_combined_anova_ML(sys_conf, "good", "random", save=True)
    for sys_conf in util_func.BAD_SYS:
        # draw_one_sys(sys_conf, util_func.get_factor_names(sys_conf), "bad", save=True)
        draw_combined_anova_ML(sys_conf, "bad", "random", save=True)
    # for method in ["random", "balance", "dist-aware"]:
    #     draw_error_multiplelines(util_func.GOOD_SYS, f"../{method}", "good", f"{method}_good", save=True)
    #     draw_error_multiplelines(util_func.BAD_SYS, f"../{method}", "bad", f"{method}_bad", save=True)

