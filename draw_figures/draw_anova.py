import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import os

def set_size(width, fraction=1):
    """Set aesthetic figure dimensions to avoid scaling in latex.

    Parameters
    ----------
    width: float
            Width in pts
    fraction: float
            Fraction of the width which you wish the figure to occupy

    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    # Width of figure
    fig_width_pt = width * fraction

    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    golden_ratio = (5**0.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio

    fig_dim = (fig_width_in, fig_height_in)

    return fig_dim


def preprocess_result(sample_method):
    input_base_dir = os.path.join(os.path.dirname(__file__), "../results/ANOVA")
    df = pd.read_csv(f"{input_base_dir}/RESULT_{sample_method}.csv.gz")
    # df = pd.read_csv(f"/ANOVA/RESULT_{sample_method}.csv")
    if sample_method == "stratified" or sample_method == "neyman":
        df.alg = df.apply(lambda x: x.alg.replace(f"_{x.stratified_term}", ""), axis=1)
    df = df[
        df.alg.isin(
            [
                "aria-Aria_tpcc",
                "calvin-Calvin_tpcc",
                "cicada-Cicada_ycsb",
                "drtm_tpcc",
                "gam_tpcc",
                "herd_ycsb",
                "janus-Janus_tpcc",
                "mysql_tpcc",
                "silo-Silo_tpcc",
                "silo_ycsb",
                "star_tpcc",
                "star_ycsb",
                "tapir_ycsb",
                "postgresql_tpcc",
                "calvin-Calvin-comp_tpcc",
                "aria-Calvin-comp_tpcc",
            ]
        )
    ]
    # ,"silo_vs_partitionedstore","janus_vs_2pl"])]
    rename_alg = {
        "aria-Aria_tpcc": "Aria-TPC-C",
        "calvin-Calvin_tpcc": "Calvin-TPC-C",
        "cicada-Cicada_ycsb": "Cicada-YCSB",
        "drtm_tpcc": "DrTM-TPC-C",
        "gam_tpcc": "GAM-TPC-C",
        "herd_ycsb": "HERD-YCSB",
        "janus-Janus_tpcc": "Janus-TPC-C",
        "mysql_tpcc": "MySQL-TPC-C",
        "silo-Silo_tpcc": "Silo-TPC-C",
        "silo_ycsb": "Silo-YCSB",
        "star_tpcc": "Star-TPC-C",
        "star_ycsb": "Star-YCSB",
        "postgresql_tpcc": "PostgreSQL-TPC-C",
        "tapir_ycsb": "TAPIR-YCSB",
        "silo_vs_partitionedstore": "Silo/PartitionedStore-TPC-C",
        "janus_vs_2pl": "Janus/2PL-TPC-C",
        "aria-Calvin-comp_tpcc": "Calvin(Aria)-TPC-C",
        "calvin-Calvin-comp_tpcc": "Calvin(original)-TPC-C",
    }
    df["alg"] = df["alg"].apply(lambda x: rename_alg[x])
    df["num_of_terms_changed"] *= 100
    df["num_turning_mismatch"] *= 100
    df["sample_ratio"] = df["sample_ratio"].apply(pd.to_numeric)
    df["random"] = df["random"].apply(pd.to_numeric)
    df["split"] = df["split"].apply(pd.to_numeric)
    df["sample_method"] = sample_method
    return df


def plot_r2():
    df = pd.concat(
        [
            preprocess_result(sample_method)
            for sample_method in ["random", "stratified", "balance", "dist-aware", "neyman"]
        ],
        ignore_index=True,
    )

    out_base = os.path.join(os.path.dirname(__file__), "../plots/ANOVA/anova_r2")
    if not os.path.exists(out_base):
        os.makedirs(out_base)

    r2_unsamples = (
        df.groupby(by=["alg", "sample_method", "sample_ratio", "random"])["r2_unsample"]
        .min()
        .reset_index()
    )

    # fig, axs = plt.subplots(4, 3, figsize=(fig_size_one[0]*3,fig_size_one[1]*4),constrained_layout=True)
    all_algs = df.alg.unique()
    x_labels = [f"{i:.2f}" for i in np.arange(0.05, 1, 0.05)]
    for idx, alg_name in enumerate(all_algs):
        # ax = axs[idx//3][idx%3]
        fig, ax = plt.subplots(1, 1, figsize=(4.78, 3.7), constrained_layout=True)
        sns.lineplot(
            data=r2_unsamples[r2_unsamples.alg == alg_name],
            x="sample_ratio",
            y="r2_unsample",
            estimator="median",
            palette="Set1",
            errorbar=("pi", 60),
            err_style="bars",
            ax=ax,
            dashes=True,
            hue="sample_method",
            style="sample_method",
            ms=1,
            mew=2,
            linewidth=1.0,
        )
        ax.set_xticks(np.arange(0.05, 1, 0.05))
        ax.set_xticklabels(x_labels, rotation=30)
        ax.set_xlabel("Sampling Rate")
        ax.set_ylabel("P50 of R2 Scores +/- to P80 and P20")
        ax.set_ylim([0, 1.01])
        ax.legend(
            loc="upper left",
            bbox_to_anchor=(0.0, 1.1, 0, 0.102),
            ncol=3,
            borderaxespad=0.0,
            numpoints=2,
        )
        fig.savefig(
            f"{out_base}/{alg_name}.pdf", bbox_inches="tight", dpi=100, format="pdf"
        )
        plt.close(fig)


def plot_random_thresholds():
    df_random = preprocess_result("random")
    all_algs = df_random.alg.unique()
    r2_thresh_to_examine = [0.2, 0.4, 0.6, 0.8, 0.9, 0.95]
    rows = []
    r2_unsamples = df_random.groupby(by=["alg", "sample_ratio", "random"])[
        "r2_unsample"
    ].min()
    for alg in all_algs:
        if alg in ["PostgreSQL-TPC-C"]:
            continue
        for random in range(10):
            for r2_thresh in r2_thresh_to_examine:
                min_r2_series = r2_unsamples.loc[alg, :, random]
                try:
                    min_sample_ratio = min_r2_series.loc[
                        min_r2_series > r2_thresh
                    ].idxmin()
                except ValueError:
                    # min_sample_ratio = min_r2_series.idxmax()
                    continue
                row = df_random.loc[
                    (df_random.alg == alg)
                    & (df_random.random == random)
                    & (df_random.sample_ratio == min_sample_ratio)
                ].copy()
                # row = row.groupby(by=["alg","sample_ratio","random"])[["num_of_terms_changed","num_turning_mismatch"]].median().reset_index()
                row["r2_thresh"] = r2_thresh
                rows.append(row)

    df_random_r2t = pd.concat(rows, ignore_index=True)
    df_random_r2t["r2_thresh"] = df_random_r2t["r2_thresh"].apply(pd.to_numeric)
    df_random_r2t["sample_ratio"] = df_random_r2t["sample_ratio"].apply(pd.to_numeric)
    df_random_r2t["num_of_terms_changed"] = df_random_r2t["num_of_terms_changed"].apply(
        pd.to_numeric
    )
    df_random_r2t["num_turning_mismatch"] = df_random_r2t["num_turning_mismatch"].apply(
        pd.to_numeric
    )

    out_base = os.path.join(
        os.path.dirname(__file__), "../plots/ANOVA/anova_metric_r2thresh"
    )
    if not os.path.exists(out_base):
        os.makedirs(out_base)

    w, h = set_size(WIDTH)
    h += 1

    for metric in ["num_turning_mismatch", "num_of_terms_changed"]:
        for is_ycsb in [True, False]:
            df_subset_idx = df_random_r2t["alg"].apply(lambda x: "YCSB" in x) == is_ycsb
            fig, ax = plt.subplots(1, 1, figsize=(w, h), constrained_layout=True)
            sns.lineplot(
                data=df_random_r2t[df_subset_idx],
                x="r2_thresh",
                y=metric,
                estimator="median",
                palette="Set1",
                errorbar=None,
                # errorbar=("pi",60),
                # err_style="bars",
                dashes=True,
                ax=ax,
                hue="alg",
                style="alg",
                markers=True,
                linewidth=1.0,
            )
            ax.set_xticks(r2_thresh_to_examine)
            ax.set_xticklabels([f"{i:.2f}" for i in r2_thresh_to_examine], rotation=30)
            ax.set_ylim([0, 105])
            ax.set_xlabel("R2 threshold")
            # ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=1, borderaxespad=0.,numpoints=2)
            if is_ycsb:
                ax.legend(
                    loc="upper left",
                    bbox_to_anchor=(0.0, 1.3),
                    ncol=2,
                    borderaxespad=0.0,
                    numpoints=2,
                )
            else:
                ax.legend(
                    loc="upper left",
                    bbox_to_anchor=(0.0, 1.4),
                    ncol=2,
                    borderaxespad=0.0,
                    numpoints=2,
                )
            if metric == "num_turning_mismatch":
                ax.set_ylabel("P50 of \%Important Points Mismatch")
            else:
                ax.set_ylabel("P50 of \%Significant Terms Changed")
            if is_ycsb:
                fig.savefig(
                    f"{out_base}/{metric}_ycsb.pdf",
                    bbox_inches="tight",
                    dpi=100,
                    format="pdf",
                )
            else:
                fig.savefig(
                    f"{out_base}/{metric}_tpc-c.pdf",
                    bbox_inches="tight",
                    dpi=100,
                    format="pdf",
                )
    plt.close(fig)


def plot_calvin_comp():
    df_calvin_comp = preprocess_result("calvin-comp")
    out_base = os.path.join(os.path.dirname(__file__), "../plots/ANOVA/anova_r2")
    if not os.path.exists(out_base):
        os.makedirs(out_base)

    r2_unsamples = (
        df_calvin_comp.groupby(by=["alg", "sample_method", "sample_ratio", "random"])[
            "r2_unsample"
        ]
        .min()
        .reset_index()
    )
    r2_unsamples = r2_unsamples[r2_unsamples.sample_ratio > 0.2]
    x_labels = [f"{i:.2f}" for i in np.arange(0.2, 1, 0.05)]
    # ax = axs[idx//3][idx%3]
    fig, ax = plt.subplots(1, 1, figsize=(4.78, 3.7), constrained_layout=True)

    sns.lineplot(
        data=r2_unsamples,
        x="sample_ratio",
        y="r2_unsample",
        estimator="median",
        palette="Set1",
        errorbar=("pi", 60),
        err_style="bars",
        ax=ax,
        dashes=True,
        hue="alg",
        style="alg",
        ms=1,
        mew=2,
        linewidth=1.0,
    )
    ax.set_xticks(np.arange(0.2, 1, 0.05))
    ax.set_xticklabels(x_labels, rotation=30)
    ax.set_ylim([0, 1.1])
    ax.set_xlabel("Sampling Rate")
    ax.set_ylabel("P50 of R2 Scores +/- to P80 and P20")
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(0.0, 1.1, 0, 0.102),
        ncol=1,
        borderaxespad=0.0,
        numpoints=2,
    )
    fig.savefig(
        f"{out_base}/calvin-comp.pdf", bbox_inches="tight", dpi=100, format="pdf"
    )
    plt.close(fig)


if __name__ == "__main__":
    # Using seaborn's style
    plt.style.use(["ggplot", "seaborn-whitegrid"])
    WIDTH = 345

    config = {
        "figure.figsize": set_size(WIDTH),
        "grid.linestyle": "--",
        "hatch.color": "#eeeeee",
        "hatch.linewidth": 0.618,
        # Use LaTeX to write all text
        "text.usetex": True,
        "font.family": "Times New Roman",
        #     "text.latex.unicode": True,
        "axes.unicode_minus": True,
        # Use 10pt font in plots, to match 10pt font in document
        "axes.labelsize": 12,
        "font.size": 12,
        "legend.fontsize": 12,
        # Make the label fonts a little smaller
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.handletextpad": 0.5,
        "legend.columnspacing": 1,
        "legend.numpoints": 3,
    }    
    mpl.rcParams.update(config)
    plt.rcParams["axes.edgecolor"] = "#333F4B"
    plt.rcParams["axes.linewidth"] = 0.8

    plot_r2()
    # plot_random_thresholds()
    # plot_calvin_comp()