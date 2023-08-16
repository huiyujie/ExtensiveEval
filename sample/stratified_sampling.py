import numpy as np

from sklearn.model_selection import train_test_split

from util import test_split


def stratified_sampling(data, factor_names, output_file=None, seed=None):
    num_seeds = 100
    samples = {}
    print(factor_names)
    if seed is not None:
        for f in factor_names:
            samples[f] = [[0] * 19] * 1
            for j, ratio in enumerate(np.linspace(0.05, 1, num=20)[:-1]):
                exp, unexp = train_test_split(data,
                                              test_size=(1 - ratio),
                                              random_state=seed,
                                              stratify=data[f])
                test_split(exp, unexp)
                if output_file:
                    exp.to_csv(f"{output_file}_{ratio:.2f}_{seed}_{f}.csv", index=False)
                samples[f][0][j] = exp
    else:
        for f in factor_names:
            samples[f] = [[0] * 19] * num_seeds
            for i in range(num_seeds):
                for j, ratio in enumerate(np.linspace(0.05, 1, num=20)[:-1]):
                    exp, unexp = train_test_split(data,
                                                  test_size=(1 - ratio),
                                                  random_state=i,
                                                  stratify=data[f])
                    test_split(exp, unexp)
                    if output_file:
                        exp.to_csv(f"{output_file}_{ratio:.2f}_{i}_{f}.csv", index=False)
                    samples[f][i][j] = exp

    return samples