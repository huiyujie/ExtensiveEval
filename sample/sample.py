import pandas as pd

from sample.balance_sampling import balance_sampling
from sample.preprocess import preprocess
from sample.random_sampling import random_sampling
from sample.stratified_sampling import stratified_sampling
from sample.dist_aware_sampling import dist_aware_sampling


def sample(data, method, system, bench, output_dir=None, seed=None, alg=None):
    data, factor_names = preprocess(system, bench, alg, data)
    if output_dir:
        if alg:
            output_file = f"{output_dir}/method@{method}_sys@{system}-{alg}_bench@{bench}"
        else:
            output_file = f"{output_dir}/method@{method}_sys@{system}_bench@{bench}"
    else:
        output_file = None

    if method == "random":
        return random_sampling(data, output_file, seed=seed)
    elif method == "stratified":
        return stratified_sampling(data, factor_names, output_file, seed=seed)
    elif method == "balance":
        return balance_sampling(data, factor_names, output_file, seed=seed)
    elif method == "dist-aware":
        return dist_aware_sampling(data, output_file, seed=seed)
    else:
        raise ValueError("Unknown sampling method")


if __name__ == "__main__":
    data = pd.read_csv("../csv/tapir_ycsb.csv")
    samples = sample(data, "balance", "tapir", "ycsb", seed=0)