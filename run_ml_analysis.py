import pandas as pd
import numpy as np
import sys, os
path_to_sample_dir = './sample'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), path_to_sample_dir)))
from ML.regressor import MLP_regression, Lasso_regression  # updated to import Lasso_regression
import sample
from util_func import ALL_SYS, get_raw_data, get_factor_names, save_r2_results, parse_arg

def run_regression(all_samples, all_unsamples, output_name, method, factor_names, regressor_type=None):
    # ML prediction
    sample_ratio = [f"{i:.2f}" for i in np.arange(0.05, 1, 0.05)]
    r2_rows = []
    for seed, samples in enumerate(all_samples):
        print(f"seed {seed}")
        for split, splitted_samples in enumerate(samples):
            for ratio, one_sample in enumerate(splitted_samples):
                if regressor_type and regressor_type.lower() == "lasso":
                    r2 = Lasso_regression(one_sample,
                                          all_unsamples[seed][split][ratio],
                                          factor_names)
                else:
                    r2 = MLP_regression(one_sample,
                                        all_unsamples[seed][split][ratio],
                                        factor_names)
                r2_rows.append({"random_seed": seed,
                                "split_seed": split,
                                "sample_ratio": sample_ratio[ratio],
                                "r2": r2})
    r2_results = pd.DataFrame(r2_rows, columns=["random_seed", "split_seed", "sample_ratio", "r2"])

    # Save results to proper directory based on regressor_type
    if regressor_type and regressor_type.lower() == "lasso":
        save_r2_results(r2_results, f"./results/Lasso/{method}", output_name)
    else:
        save_r2_results(r2_results, f"./results/ML/{method}", output_name)

if __name__ == "__main__":
    args = parse_arg()
    print(args)

    # Determine which systems to run
    if args.all:
        systems_to_run = list(range(len(ALL_SYS)))
    else:
        if args.systems is None:
            raise ValueError("Please specify --systems indices or use --all flag to run all systems")
        systems_to_run = args.systems

    # Use the new argument '--regressor' if available, defaulting to None
    regressor = getattr(args, "regressor", None)

    for index in systems_to_run:
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
                run_regression(all_samples[term], all_unsamples[term], output_name, method, factor_names, regressor)
        else:
            output_name = f"{system}-{bench}-{alg}.csv" if alg else f"{system}-{bench}.csv"
            run_regression(all_samples, all_unsamples, output_name, method, factor_names, regressor)