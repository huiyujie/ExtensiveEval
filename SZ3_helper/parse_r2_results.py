import numpy as np
import pandas as pd

import util_func

def find_ratio(dataframe):
    mean_values = dataframe.iloc[:, 1:].mean(axis=1)
    filtered = mean_values[mean_values >= 0.9]
    if len(filtered) > 0:
        idx = filtered.idxmin()
    else:
        return 1
    ratio = dataframe.iloc[idx, 0]
    return ratio

def read_r2_scores(filename):
    file_dir = "random/" + filename + ".csv"
    data = pd.read_csv(file_dir)

    ratio = find_ratio(data)
    return ratio

if __name__ == "__main__":
    ratios = {}
    for comb in util_func.ALL_SYS:
        filename = comb[0]+"-"+comb[1]+"-"+comb[2] if len(comb)==3 else comb[0]+"-"+comb[1]
        r = read_r2_scores(filename)
        ratios[filename] = r

    # ratios = dict(sorted(ratios.items(), key=lambda item: item[1]))
    for k in ratios.keys():
        print(f"{k},{ratios[k]}")
