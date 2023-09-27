import numpy as np

from sklearn.model_selection import train_test_split

from util import test_split, split_sampleset

'''
If no seed provided:
 Return samples[num_seed][split][19]
If seed provided:
 Return samples[1][split][19]
'''
def random_sampling(data, output_file=None, seed=None):
    num_seeds = 10
    split = 10
    if seed is not None:
        samples = [[[0] * 19 for _ in range(split)] for _ in range(1)]
        unsamples = [[[0] * 19 for _ in range(split)] for _ in range(1)]
        for j, ratio in enumerate(np.linspace(0.05, 1, num=20)[:-1]):
            exp, unexp = train_test_split(data, test_size=(1 - ratio), random_state=seed)
            test_split(exp, unexp)
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
                exp, unexp = train_test_split(data, test_size=(1 - ratio), random_state=i)
                test_split(exp, unexp)
                for r_split in range(split):
                    sample_exp, sample_unexp = split_sampleset(exp,r_split)
                    if output_file:
                        sample_exp.to_csv(f"{output_file}_sample@{ratio:.2f}_random@{i}_split@{r_split}_exp.csv", index=False)
                        sample_unexp.to_csv(f"{output_file}_sample@{ratio:.2f}_random@{i}_split@{r_split}_unexp.csv", index=False)
                    samples[i][r_split][j] = sample_exp
                    unsamples[i][r_split][j] = sample_unexp

    return samples, unsamples