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
    # num_seeds = 100
    num_seeds = 10
    if seed is not None:
        samples = [[0] * 19 for _ in range(10)]
        unsamples = [[0] * 19 for _ in range(10)]
        for j, ratio in enumerate(np.linspace(0.05, 1, num=20)[:-1]):
            exp, unexp = train_test_split(data, test_size=(1 - ratio), random_state=seed)
            test_split(exp, unexp)
            for r_split in range(10):
                sample_exp, sample_unexp = split_sampleset(exp,r_split)
                if output_file:
                    sample_exp.to_csv(f"{output_file}_sample@{ratio:.2f}_random@{seed}_split@{r_split}_exp.csv", index=False)
                    sample_unexp.to_csv(f"{output_file}_sample@{ratio:.2f}_random@{seed}_split@{r_split}_unexp.csv", index=False)
                samples[r_split][j] = sample_exp
                unsamples[r_split][j] = sample_unexp
    else:
        samples = [[0] * 19 for _ in range(num_seeds*10)]
        unsamples = [[0] * 19 for _ in range(num_seeds*10)]
        for i in range(num_seeds):
            for j, ratio in enumerate(np.linspace(0.05, 1, num=20)[:-1]):
                exp, unexp = train_test_split(data, test_size=(1 - ratio), random_state=i)
                test_split(exp, unexp)
                for r_split in range(10):
                    sample_exp, sample_unexp = split_sampleset(exp,r_split)
                    if output_file:
                        sample_exp.to_csv(f"{output_file}_sample@{ratio:.2f}_random@{i}_split@{r_split}_exp.csv", index=False)
                        sample_unexp.to_csv(f"{output_file}_sample@{ratio:.2f}_random@{i}_split@{r_split}_unexp.csv", index=False)
                    samples[i*10+r_split][j] = sample_exp
                    unsamples[i*10+r_split][j] = sample_unexp

    return samples, unsamples