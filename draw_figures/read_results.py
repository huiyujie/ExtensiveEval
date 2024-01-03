import pandas as pd
import numpy as np

def get_results_old_format(filename):
    data = pd.read_csv(filename)
    data["nan_cnt"] = data.isna().sum(axis=1)
    data["p20"] = data.iloc[:, 1:].quantile(q=0.2, axis=1, interpolation="nearest")
    data["p50"] = data.iloc[:, 1:].quantile(q=0.5, axis=1, interpolation="nearest")
    data["p80"] = data.iloc[:, 1:].quantile(q=0.8, axis=1, interpolation="nearest")

    data.loc[data["nan_cnt"] > 50, :] = np.nan

    return data

def get_results(filename):
    data = pd.read_csv(filename)
    data = data.pivot_table(index="sample_ratio",
                            columns="random_seed",
                            values="r2",
                            aggfunc="min")
    data = data.reset_index()
    data = data.rename(columns={"sample_ratio": "ratio"})
    data["p20"] = data.iloc[:, 1:].quantile(q=0.2, axis=1, interpolation="nearest")
    data["p50"] = data.iloc[:, 1:].quantile(q=0.5, axis=1, interpolation="nearest")
    data["p80"] = data.iloc[:, 1:].quantile(q=0.8, axis=1, interpolation="nearest")
    return data


def get_anova_results(filename, sample_method):
    df = pd.read_csv(f"../results/ANOVA/RESULT_{sample_method}.csv")
    if sample_method == "stratified":
        df.alg = df.apply(lambda x: x.alg.replace(f"_{x.stratified_term}",""), axis=1)
    df = df[df.alg.isin(['aria-Aria_tpcc', 'calvin-Calvin_tpcc', 'cicada-Cicada_ycsb',
        'drtm_tpcc', 'gam_tpcc', 'herd_ycsb', 'janus-Janus_tpcc', 'mysql_tpcc',
        'silo-Silo_tpcc', 'silo_ycsb', 'star_tpcc', 'star_ycsb','tapir_ycsb', 'postgresql_tpcc'])]
    rename_alg = {
        'aria-Aria_tpcc': "aria-tpcc-Aria",
        'calvin-Calvin_tpcc': "calvin-tpcc-Calvin",
        'cicada-Cicada_ycsb': "cicada-ycsb-Cicada",
        'drtm_tpcc': "drtm-tpcc",
        'gam_tpcc': "gam-tpcc",
        'herd_ycsb': "herd-ycsb",
        'janus-Janus_tpcc': "janus-tpcc-Janus",
        'mysql_tpcc': "mysql-tpcc",
        'silo-Silo_tpcc': "silo-tpcc-Silo",
        'silo_ycsb': "silo-ycsb",
        'star_tpcc': "star-tpcc",
        'star_ycsb': "star-ycsb",
        'tapir_ycsb': 'tapir-ycsb',
        'postgresql_tpcc': 'postgresql-tpcc',}

    df['alg'] = df['alg'].apply(lambda x: rename_alg[x])
    data = df[df.alg == filename]
    pivot_df = data.pivot_table(index="sample_ratio",
                                columns="random",
                                values="r2_unsample",
                                aggfunc="min")
    pivot_df = pivot_df.reset_index()
    pivot_df = pivot_df.rename(columns={"sample_ratio":"ratio"})
    pivot_df["p20"] = pivot_df.iloc[:, 1:].quantile(q=0.2, axis=1, interpolation="nearest")
    pivot_df["p50"] = pivot_df.iloc[:, 1:].quantile(q=0.5, axis=1, interpolation="nearest")
    pivot_df["p80"] = pivot_df.iloc[:, 1:].quantile(q=0.8, axis=1, interpolation="nearest")

    missing_rows = 19 - pivot_df.shape[0]
    if missing_rows > 0:
        empty_data = {col: [None] * missing_rows for col in pivot_df.columns}
        empty_rows_df = pd.DataFrame(empty_data)
        pivot_df = pd.concat([empty_rows_df, pivot_df], ignore_index=True)

    return pivot_df