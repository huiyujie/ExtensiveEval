import pandas as pd
import numpy as np
import sys,os

from ExtensiveEval.run_ml_analysis import run_regression


path_to_sample_dir = './sample'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), path_to_sample_dir)))
from ML.regressor import MLP_regression
import sample
from calvin_comparison import read_calvin_data
from random_sampling import random_sampling


def calvin_comparison(method):
    original_calvin, aria_calvin = read_calvin_data()
    factor_names = ["WH", "dist"]

    original_all_samples, original_all_unsamples = random_sampling(original_calvin)
    aria_calvin_all_samples, aria_calvin_all_unsamples = random_sampling(aria_calvin)

    run_regression(original_all_samples, original_all_unsamples, "calvin_original.csv", method, factor_names)
    run_regression(aria_calvin_all_samples, aria_calvin_all_unsamples, "calvin_aria.csv", method, factor_names)

if __name__ == "__main__":
    calvin_comparison("random")