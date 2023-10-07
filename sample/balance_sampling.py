import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd

from util import test_split, split_sampleset


def balance_sampling(data, factor_names, output_file=None, seed=None):
    print("balance sampling")
    num_seeds = 100
    split = 10
    if seed is not None:
        samples = [[[0] * 19 for _ in range(split)] for _ in range(1)]
        unsamples = [[[0] * 19 for _ in range(split)] for _ in range(1)]
        for j, ratio in enumerate(np.linspace(0.05, 1, num=20)[:-1]):
            if j == 0:
                exp, unexp = train_test_split(data, test_size=0.95, random_state=seed)
            else:
                data_tobe_added = len(data) * ratio - len(exp)
                exp, unexp = balance_data(exp, unexp, factor_names, data_tobe_added)
            for r_split in range(split):
                sample_exp, sample_unexp = split_sampleset(exp, r_split)
                if output_file:
                    sample_exp.to_csv(f"{output_file}_sample@{ratio:.2f}_random@{seed}_split@{r_split}_exp.csv", index=False)
                    sample_unexp.to_csv(f"{output_file}_sample@{ratio:.2f}_random@{seed}_split@{r_split}_unexp.csv", index=False)
                samples[0][r_split][j] = sample_exp
                unsamples[0][r_split][j] = sample_unexp
    else:
        samples = [[[0] * 19 for _ in range(split)] for _ in range(num_seeds)]
        unsamples = [[[0] * 19 for _ in range(split)] for _ in range(num_seeds)]
        for i in range(num_seeds):
            for j, ratio in enumerate(np.linspace(0.05, 1, num=20)[:-1]):
                if j == 0:
                    exp, unexp = train_test_split(data, test_size=0.95, random_state=i)
                else:
                    data_tobe_added = len(data) * ratio - len(exp)
                    exp, unexp = balance_data(exp, unexp, factor_names, data_tobe_added)
                for r_split in range(split):
                    sample_exp, sample_unexp = split_sampleset(exp, r_split)
                    if output_file:
                        sample_exp.to_csv(f"{output_file}_sample@{ratio:.2f}_random@{i}_split@{r_split}_exp.csv",
                                          index=False)
                        sample_unexp.to_csv(f"{output_file}_sample@{ratio:.2f}_random@{i}_split@{r_split}_unexp.csv",
                                            index=False)
                    samples[i][r_split][j] = sample_exp
                    unsamples[i][r_split][j] = sample_unexp

    return samples, unsamples

def balance_data(train, test, param_names, cnt):
    train = train.copy(deep=True)
    test = test.copy(deep=True)

    while cnt > 0 and len(test) != 0:
        remove_df = test
        for para in param_names:
            #TODO: special case for tapir_ycsb due to
            # these two factors are correlated
            if para == "nShard" or para == "nrep": continue
            fre = -1
            tmp_df = remove_df[remove_df[para] == find_lowest_value(train, remove_df, para, fre)]
            while len(tmp_df) == 0:
                fre = fre - 1
                tmp_df = remove_df[remove_df[para] == find_lowest_value(train, remove_df, para, fre)]
            remove_df = tmp_df
        remove_df = remove_df.sample()
        test.drop(remove_df.index, inplace=True)
        train = pd.concat([train, remove_df])
        # df.append is deprecated in pandas 2.x
        # train = train.append(remove_df)
        cnt = cnt - 1
        param_names = rotate(param_names, 1)
    test_split(train, test)
    return train, test

#fre: -1 is lowest, -2, -3 .... 0 is highest
def find_lowest_value(train, test, factor, fre):
    # -1:lowest 0:highest
    values_in_train = train[factor].unique()
    values_in_test = test[factor].unique()
    mask = np.isin(values_in_test, values_in_train) # check wether train has all values in test
    if len(values_in_test[~mask]) != 0:
        return values_in_test[~mask][0]
    return train[factor].value_counts().index[fre]

def rotate(l, n):
    return l[n:] + l[:n]