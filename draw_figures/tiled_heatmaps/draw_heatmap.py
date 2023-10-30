import pandas as pd
import numpy as np
from common import *


def plot_aria():
    PAPER = "aria"
    if not os.path.isdir("./out/{}".format(PAPER)):
        os.makedirs("./out/{}".format(PAPER))
    if not os.path.isdir("./cache/{}".format(PAPER)):
        os.makedirs("./cache/{}".format(PAPER))
    df = pd.read_csv("in/aria.csv")
    df.loc[df.Throughput < 0, "Throughput"] = np.nan
    df["TPperTHD"] = df["Throughput"] / df["#Threads"]
    tput_col_name = "TPperTHD"
    rep_dict = {"#WH": "#WH", "#Distributed": "%Cross-WH", "#Threads": "#Threads"}
    y_subax_loc = [-1, -2]
    x_subax_loc = [-3, -4]
    fontsize = 16
    column_names_exclude = ["Alg", "Throughput", "TPperTHD"]

    # gen_cmap(df,"cicada_ycsb",y_subax_loc,x_subax_loc)
    for alg_name in ["Aria"]:
        dfa = df.loc[df.Alg == alg_name].copy()
        fig_name = "{}_tpcc".format(alg_name)
        gen_cmap(
            dfa,
            fig_name,
            column_names_exclude,
            rep_dict,
            y_subax_loc,
            x_subax_loc,
            PAPER,
            tput_col_name=tput_col_name,
            fontsize=fontsize,
        )


def plot_calvin():
    PAPER = "calvin"
    if not os.path.isdir("./out/{}".format(PAPER)):
        os.makedirs("./out/{}".format(PAPER))
    if not os.path.isdir("./cache/{}".format(PAPER)):
        os.makedirs("./cache/{}".format(PAPER))
    df = pd.read_csv("in/calvin.csv")
    df["TPperTHD"] = df["tp"] / df["nnodes"] / 4
    df.loc[df.tp < 0, "tp"] = -1
    tput_col_name = "TPperTHD"
    column_names_exclude = ["alg", "tp", "TPperTHD"]
    rep_dict = {"nnodes": "#Nodes", "#WH": "#WH", "dist": "%Cross-WH"}
    y_subax_loc = [-1, -2]
    x_subax_loc = [-3, -4]
    fontsize = 16

    for alg_name in ["Calvin"]:
        dfa = df.loc[df.alg == alg_name].copy()
        dfa = dfa.loc[dfa["tp"] < 1e8]
        fig_name = "{}_tpcc".format(alg_name)
        gen_cmap(
            dfa,
            fig_name,
            column_names_exclude,
            rep_dict,
            y_subax_loc,
            x_subax_loc,
            PAPER,
            tput_col_name=tput_col_name,
            fontsize=fontsize,
        )


def plot_drtm():
    PAPER = "drtm"
    if not os.path.isdir("./out/{}".format(PAPER)):
        os.makedirs("./out/{}".format(PAPER))
    if not os.path.isdir("./cache/{}".format(PAPER)):
        os.makedirs("./cache/{}".format(PAPER))
    df = pd.read_csv("in/drtm_tpcc.csv")
    df = df.loc[df.nthread != 32]
    df["avgthd"] = df["avg"] / df["node"] / df["nthread"]
    tput_col_name = "avgthd"
    rep_dict = {
        "node": "#nodes",
        "nthread": "#Threads",
        "scale_factor": "#WH",
        "cross_warehouse": "%Cross-WH",
    }
    y_subax_loc = [-2, -4]
    x_subax_loc = [-4, -6]
    fontsize = 16
    column_names_exclude = ["iter 1", "iter 2", "iter 3", "avg", "avgthd"]

    fig_name = "drtm_tpcc"
    gen_cmap(
        df,
        fig_name,
        column_names_exclude,
        rep_dict,
        y_subax_loc,
        x_subax_loc,
        PAPER,
        tput_col_name=tput_col_name,
        fontsize=fontsize,
    )


def plot_mysql():
    tput_col_name = "tp"
    PAPER = "mysql"
    if not os.path.isdir("./out/{}".format(PAPER)):
        os.makedirs("./out/{}".format(PAPER))
    if not os.path.isdir("./cache/{}".format(PAPER)):
        os.makedirs("./cache/{}".format(PAPER))
    df = pd.read_csv("in/mysql-ram-tpcc.csv")
    df["tp"] = df["tp"] / df["terminal"]
    rep_dict = {
        "WH": "#WH",
        "terminal": "#terminal",
        "num_chunk": "#mem_pool_chunk",
        "chunk_size": "mem_pool_chunk_size(MB)",
    }
    y_subax_loc = [-2, -3, -4]
    x_subax_loc = [-4, -8, -12]
    fontsize = 16
    column_names_exclude = ["tp"]

    fig_name = "mysql_tpcc"
    gen_cmap(
        df,
        fig_name,
        column_names_exclude,
        rep_dict,
        y_subax_loc,
        x_subax_loc,
        PAPER,
        tput_col_name=tput_col_name,
        fontsize=fontsize,
    )


if __name__ == "__main__":
    plot_aria()
    plot_calvin()
    plot_drtm()
    plot_mysql()
