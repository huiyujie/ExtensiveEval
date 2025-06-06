import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from util import test_split, split_sampleset

def neyman_sampling(data, factor_names, output_file=None, seed=None):
    """
    Neyman allocation sampling on `data` stratified by a list of factors.
    `factor_names` is a list of column names. Returns two dicts mapping each factor
      to its samples and held‐out sets with structure [num_seeds][splits][num_ratios].
    """
    # total number of random seeds to iterate (unless a single seed is provided)
    num_seeds = 100
    # how many cross‐validation splits to perform on each sampled set
    splits = 10
    # range of sample ratios (e.g., 5%, 10%, …, 95%)
    ratios = np.linspace(0.05, 1, num=20)[:-1]
    # use a single seed if provided, else range(0..99)
    seeds = [seed] if seed is not None else range(num_seeds)

    all_samples = {}    # dict: factor -> sampled training sets
    all_unsamples = {}  # dict: factor -> held‐out sets

    # iterate over each stratification factor
    for f in factor_names:
        samples_f, unsamples_f = [], []
        # loop over each random seed
        for i in seeds:
            # prepare storage for this seed: splits × ratios
            samp_by_split = [[None] * len(ratios) for _ in range(splits)]
            unsamp_by_split = [[None] * len(ratios) for _ in range(splits)]
            rng = np.random.default_rng(i)  # reproducible RNG

            # iterate each desired sample ratio
            for j, ratio in enumerate(ratios):
                # 1) compute how many total points to sample
                total_n = int(np.floor(len(data) * ratio))

                # 2) group data by the stratification factor
                grp = data.groupby(f)
                # number of units in each stratum
                N_h = grp.size()
                # standard deviation of target variable "tput" per stratum
                S_h = grp['tput'].std().fillna(0)

                # 3) compute Neyman weights ∝ N_h * S_h
                weight = N_h * S_h
                # fraction of total sample for each stratum
                alloc_frac = weight / weight.sum()

                # 4) initial integer allocation by flooring expected counts
                n_h = np.floor(alloc_frac * total_n).astype(int)

                # 5) distribute any remainder (due to flooring) to strata
                rem = total_n - n_h.sum()
                if rem > 0:
                    # fractional remainders
                    frac = alloc_frac * total_n - n_h
                    # give one extra to the rem strata with largest fractional parts
                    for idx in np.argsort(-frac.values)[:rem]:
                        n_h.iloc[idx] += 1

                # 6) draw samples within each stratum according to n_h
                sampled_parts = []
                leftover_parts = []
                for stratum, nh in n_h.items():
                    block = data[data[f] == stratum]
                    # cap sample count to available data to prevent oversampling error
                    nh_to_sample = min(nh, len(block))
                    if nh_to_sample > 0:
                        # sample nh points reproducibly
                        samp = block.sample(n=nh_to_sample, random_state=i)
                    else:
                        # no samples: create empty DataFrame
                        samp = block.iloc[0:0]
                    # the rest of the block not sampled
                    rest = block.drop(samp.index)
                    sampled_parts.append(samp)
                    leftover_parts.append(rest)

                # combine all strata
                exp = pd.concat(sampled_parts)    # sampled data
                unexp = pd.concat(leftover_parts) # hold‐out data
                # optional tagging or bookkeeping
                test_split(exp, unexp)

                # 7) within this sampled set, perform multiple train/test splits
                for r in range(splits):
                    e_train, e_test = split_sampleset(exp, r)
                    if output_file:
                        # write split to disk if requested
                        e_train.to_csv(
                            f"{output_file}_sample@{ratio:.2f}_random@{i}_term@{f}_split@{r}_exp.csv",
                            index=False
                        )
                        e_test.to_csv(
                            f"{output_file}_sample@{ratio:.2f}_random@{i}_term@{f}_split@{r}_unexp.csv",
                            index=False
                        )
                    # store in our per‐seed matrix
                    samp_by_split[r][j] = e_train
                    unsamp_by_split[r][j] = e_test
            samples_f.append(samp_by_split)
            unsamples_f.append(unsamp_by_split)
        if seed is not None:
            all_samples[f] = [samples_f[0]]
            all_unsamples[f] = [unsamples_f[0]]
        else:
            all_samples[f] = samples_f
            all_unsamples[f] = unsamples_f
    return all_samples, all_unsamples