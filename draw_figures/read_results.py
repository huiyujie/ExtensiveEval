import pandas as pd
import numpy as np

def get_results(filename):
    data = pd.read_csv(filename)
    data["nan_cnt"] = data.isna().sum(axis=1)
    data["p20"] = data.iloc[:, 1:].quantile(q=0.2, axis=1, interpolation="nearest")
    data["p50"] = data.iloc[:, 1:].quantile(q=0.5, axis=1, interpolation="nearest")
    data["p80"] = data.iloc[:, 1:].quantile(q=0.8, axis=1, interpolation="nearest")

    data.loc[data["nan_cnt"] > 50, :] = np.nan

    return data