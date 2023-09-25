import numpy as np

from sklearn.model_selection import train_test_split

from util import test_split, split_sampleset

'''
If no seed provided:
 Return samples[100][19]
If seed provided:
 Return samples[1][19]
'''
def random_sampling(data, output_file=None, seed=None):
    num_seeds = 100
    if seed is not None:
        samples = [[0] * 19 for _ in range(1)]
        unsamples = [[0] * 19 for _ in range(1)]
        for j, ratio in enumerate(np.linspace(0.05, 1, num=20)[:-1]):
            exp, unexp = train_test_split(data, test_size=(1 - ratio), random_state=seed)
            test_split(exp, unexp)
            sample_exp, sample_unexp = split_sampleset(exp)
            if output_file:
                sample_exp.to_csv(f"{output_file}_sample@{ratio:.2f}_random@{seed}_exp.csv", index=False)
                sample_unexp.to_csv(f"{output_file}_sample@{ratio:.2f}_random@{seed}_unexp.csv", index=False)
            samples[0][j] = sample_exp
            unsamples[0][j] = sample_unexp
    else:
        samples = [[0] * 19 for _ in range(num_seeds)]
        unsamples = [[0] * 19 for _ in range(num_seeds)]
        for i in range(num_seeds):
            for j, ratio in enumerate(np.linspace(0.05, 1, num=20)[:-1]):
                exp, unexp = train_test_split(data, test_size=(1 - ratio), random_state=i)
                test_split(exp, unexp)
                sample_exp, sample_unexp = split_sampleset(exp)
                if output_file:
                    sample_exp.to_csv(f"{output_file}_sample@{ratio:.2f}_random@{i}_exp.csv", index=False)
                    sample_unexp.to_csv(f"{output_file}_sample@{ratio:.2f}_random@{i}_unexp.csv", index=False)
                samples[i][j] = sample_exp
                unsamples[i][j] = sample_unexp

    return samples, unsamples