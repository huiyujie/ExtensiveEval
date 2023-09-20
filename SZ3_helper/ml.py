import sys,os


path_to_dir = '../ML'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), path_to_dir)))
from regressor import MLP_regression
from parse_results import get_global_index
import pandas as pd
pd.set_option('display.max_columns', None)

def ml(original, dim_info, samples):
    predicting_data = set_samples(original, samples, dim_info.get_dimensions())
    factor_names = dim_info.get_factor_names()
    train, valid = extract_sample_train(predicting_data)
    train.drop(columns=["is_test", "is_filled", "_merge"], inplace=True)
    valid.drop(columns=["is_test", "is_filled", "_merge"], inplace=True)
    r2 = MLP_regression(train, valid, factor_names)
    return r2


def set_samples(predict_data, samples, dims):
    predict_data["is_test"] = True
    for sample in samples:
        offset = get_global_index(sample, dims)
        predict_data.loc[offset, "is_test"] = False

    return predict_data

def extract_sample_train(predict_data):
    predict_data = predict_data[predict_data["is_filled"] == False]
    train = predict_data[predict_data["is_test"] == False]
    valid = predict_data[predict_data["is_test"] == True]

    return train, valid