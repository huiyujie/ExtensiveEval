import pandas as pd
import numpy as np
import sys,os
path_to_sample_dir = './sample'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), path_to_sample_dir)))
from ML.regressor import MLP_regression
import sample
from util_func import ALL_SYS, get_raw_data, get_factor_names, save_r2_results, parse_arg

def run_regression(all_samples, all_unsamples, output_name, method, factor_names):
    # ML prediction
    r2_results = np.empty((19, 100))
    r2_results[:] = np.nan
    for i, samples in enumerate(all_samples):
        print(f"seed {i}")
        for j, one_sample in enumerate(samples):
            r2 = MLP_regression(one_sample, all_unsamples[i][j], factor_names)
            r2_results[j][i] = r2

    # Save results
    save_r2_results(r2_results, f"./results/ML/{method}", output_name)

if __name__ == "__main__":
    args = parse_arg()
    print(args)

    for index in args.systems:
        system_conf = ALL_SYS[index]
        system = system_conf[0]
        bench = system_conf[1]
        alg = system_conf[2] if len(system_conf) == 3 else None
        method = args.method
        # Read data
        data = get_raw_data(system_conf)
        factor_names = get_factor_names(system_conf)

        # Generate all samples
        all_samples, all_unsamples = sample.sample(data, method, system=system, bench=bench, alg=alg)

        # Run regression
        if method == "stratified":
            for term in factor_names:
                output_name = f"{system}-{bench}-{alg}-{term}.csv" if alg else f"{system}-{bench}-{term}.csv"
                run_regression(all_samples[term], all_unsamples[term], output_name, method, factor_names)
        else:
            output_name = f"{system}-{bench}-{alg}.csv" if alg else f"{system}-{bench}.csv"
            run_regression(all_samples, all_unsamples, output_name, method, factor_names)
