import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd

from util import test_split


def balance_sampling(data, factor_names, output_file=None, seed=None):
    print("balance sampling")
    num_seeds = 100
    if seed is not None:
        samples = [[0] * 19] * 1
        unsamples = [[0] * 19] * 1
        for j, ratio in enumerate(np.linspace(0.05, 1, num=20)[:-1]):
            if j == 0:
                exp, unexp = train_test_split(data, test_size=0.95, random_state=seed)
                samples[0][j] = exp
                unsamples[0][j] = unexp
            else:
                data_tobe_added = len(data) * ratio - len(samples[0][j-1])
                exp, unexp = balance_data(exp, unexp, factor_names, data_tobe_added)
                samples[0][j] = exp
                unsamples[0][j] = unexp
            if output_file:
                exp.to_csv(f"{output_file}_sample@{ratio:.2f}_random@{seed}_exp.csv", index=False)
                unexp.to_csv(f"{output_file}_sample@{ratio:.2f}_random@{seed}_unexp.csv", index=False)
    else:
        samples = [[0] * 19] * num_seeds
        unsamples = [[0] * 19] * num_seeds
        for i in range(num_seeds):
            for j, ratio in enumerate(np.linspace(0.05, 1, num=20)[:-1]):
                if j == 0:
                    exp, unexp = train_test_split(data, test_size=0.95, random_state=i)
                    samples[i][j] = exp
                    unsamples[i][j] = unexp
                else:
                    data_tobe_added = len(data) * ratio - len(samples[i][j-1])
                    exp, unexp = balance_data(exp, unexp, factor_names, data_tobe_added)
                    samples[i][j] = exp
                    unsamples[i][j] = unexp
                if output_file:
                    exp.to_csv(f"{output_file}_sample@{ratio:.2f}_random@{i}_exp.csv", index=False)
                    unexp.to_csv(f"{output_file}_sample@{ratio:.2f}_random@{i}_unexp.csv", index=False)

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