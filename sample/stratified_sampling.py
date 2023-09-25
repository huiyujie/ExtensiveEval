import numpy as np

from sklearn.model_selection import train_test_split

from util import test_split, split_sampleset


def stratified_sampling(data, factor_names, output_file=None, seed=None):
    num_seeds = 100
    samples = {}
    unsamples = {}
    print(factor_names)
    if seed is not None:
        for f in factor_names:
            samples[f] = [[0] * 19 for _ in range(1)]
            unsamples[f] = [[0] * 19 for _ in range(1)]
            for j, ratio in enumerate(np.linspace(0.05, 1, num=20)[:-1]):
                exp, unexp = train_test_split(data,
                                              test_size=(1 - ratio),
                                              random_state=seed,
                                              stratify=data[f])
                test_split(exp, unexp)
                sample_exp, sample_unexp = split_sampleset(exp)
                if output_file:
                    sample_exp.to_csv(f"{output_file}_sample@{ratio:.2f}_random@{seed}_term@{f}_exp.csv", index=False)
                    sample_unexp.to_csv(f"{output_file}_sample@{ratio:.2f}_random@{seed}_term@{f}_unexp.csv", index=False)
                samples[f][0][j] = sample_exp
                unsamples[f][0][j] = sample_unexp
    else:
        for f in factor_names:
            samples[f] = [[0] * 19 for _ in range(num_seeds)]
            unsamples[f] = [[0] * 19 for _ in range(num_seeds)]
            for i in range(num_seeds):
                for j, ratio in enumerate(np.linspace(0.05, 1, num=20)[:-1]):
                    exp, unexp = train_test_split(data,
                                                  test_size=(1 - ratio),
                                                  random_state=i,
                                                  stratify=data[f])
                    test_split(exp, unexp)
                    sample_exp, sample_unexp = split_sampleset(exp)
                    if output_file:
                        sample_exp.to_csv(f"{output_file}_sample@{ratio:.2f}_random@{seed}_term@{f}_exp.csv",
                                          index=False)
                        sample_unexp.to_csv(f"{output_file}_sample@{ratio:.2f}_random@{seed}_term@{f}_unexp.csv",
                                            index=False)
                    samples[f][i][j] = sample_exp
                    unsamples[f][i][j] = sample_unexp

    return samples, unsamples