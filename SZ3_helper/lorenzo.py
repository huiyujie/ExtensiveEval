import copy
import pandas as pd
from sklearn.metrics import r2_score

from parse_results import get_global_index




def lorenzo(original, dim_info, samples):
    predicting_data = set_samples(original, samples, dim_info.get_dimensions())
    predict(predicting_data, dim_info)
    predicted_rows = predicting_data[(predicting_data['is_filled'] == False) & (predicting_data['is_test'] == True)]
    gt = original.loc[predicted_rows.index]
    r2 = r2_score(gt["tput"], predicted_rows["tput"])

    return r2

def set_samples(dataframe, samples, dims):
    predict_data = dataframe.copy(deep=True)
    predict_data["tput"] = float('nan')
    for sample in samples:
        offset = get_global_index(sample, dims)
        predict_data.loc[offset, "tput"] = dataframe.loc[offset, "tput"]

    predict_data["is_test"] = predict_data["tput"].isna()
    return predict_data


def predict(predicting_data, dim_info):
    for index, row in predicting_data.iterrows():
        if not pd.isna(row["tput"]):
            # print(row)
            continue
        predicted = predict_one(predicting_data, row, dim_info)
        predicting_data.loc[index, "tput"] = predicted

def predict_one(predicting_data, row, dim_info):
    predicted = 0
    dim = dim_info.get_dim_len()
    for n in range(1, 2**dim):
        str = f"{n:0{dim}b}"
        cnt = str.count("1")
        indices = [int(char) for char in str]
        if cnt % 2 == 0:
            predicted -= get_prev(predicting_data, row, indices, dim_info)
        else:
            predicted += get_prev(predicting_data, row, indices, dim_info)
    return predicted

def get_prev(data, row, indices, dim_info):
    for d in range(dim_info.get_dim_len()):
        cur_value = row[dim_info.param_names[d]]
        if indices[d] == 0:
            target_value = cur_value
        else:
            levels = dim_info.get_levels_by_id(d)
            idx = levels.index(cur_value)
            if idx == 0: return 0
            target_value = levels[idx - 1]
        data = data.loc[data[dim_info.param_names[d]] == target_value]
    if len(data) == 0: return 0 # It could be an empty data due to non-factorial dataset
    ret = float(data["tput"])
    if pd.isna(ret): return 0
    return ret





