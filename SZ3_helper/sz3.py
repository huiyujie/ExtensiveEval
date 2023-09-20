import os
import sys

import ml


path_to_dir = '../ML'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), path_to_dir)))

from dim_info import DimInfo
from lorenzo import lorenzo
from anova import anova
from parse_results import parse_sz3_results, get_global_index, parse_unpred_data
from data_construct import insert_missing_data, save_tput
from data_convert import compile_converter, run_converter
from run_sz3 import generate_command, run_sz3

from sample.preprocess import preprocess

sys.path.append("..")
import util_func

if __name__ == "__main__":
    args = util_func.parse_arg()
    print(args)
    if args.all:
        system_indexs = range(len(util_func.ALL_SYS))
    else:
        system_indexs = args.systems

    compile_converter()
    r2_results = {}
    ml_r2_results = {}
    anova_r2_results = {}
    sample_ratios = {}

    # set relative error bound

    rel_bounds = ["0.5", "0.45", "0.4", "0.35", "0.3", "0.25", "0.2", "0.15", "0.1", "0.05"]

    for rel in rel_bounds:
        os.environ["REL"] = rel
        r2_results[rel] = {}
        ml_r2_results[rel] = {}
        anova_r2_results[rel] = {}
        sample_ratios[rel] = {}
        # Read data
        for index in system_indexs:
            system_conf = util_func.ALL_SYS[index]
            system = system_conf[0]
            bench = system_conf[1]
            alg = system_conf[2] if len(system_conf) == 3 else None
            data = util_func.get_raw_data(system_conf)
            data, _ = preprocess(system, bench, alg, data)
            factor_names = util_func.get_factor_names(system_conf)

            # Generate full factorial data by filling missing data as 0
            full_data, dimensions = insert_missing_data(data, factor_names)
            dim_info = DimInfo(full_data, factor_names)
            output_name = f"{system}-{bench}-{alg}" if alg else f"{system}-{bench}"
            save_tput(full_data, "./tput_txt", f"{output_name}.txt")

            # run converter.cpp
            util_func.check_dir("./tput_dat")
            run_converter(f"./tput_txt/{output_name}.txt", f"./tput_dat/{output_name}.dat")

            # run sz3.cpp
            commands = generate_command(dimensions, output_name)
            print(" ".join(commands))
            msg = run_sz3(commands)
            # print(msg)
            samples = parse_unpred_data(msg)
            sample_ratio, r2 = parse_sz3_results(msg, output_name)
            r2_results[rel][output_name] = r2
            sample_ratios[rel][output_name] = sample_ratio

            # Conduct Lorenzo predict
            # r2 = lorenzo(full_data, dim_info, samples)
            # print(f"r2(LORENZO): {r2}")

            # Conduct ML predict
            r2 = ml.ml(full_data, dim_info, samples)
            ml_r2_results[rel][output_name] = r2
            print(f"r2(ML): {r2}")

            # Conduct ANOVA predict
            r2 = anova(full_data, dim_info, samples)
            anova_r2_results[rel][output_name] = r2
            print(f"r2(ANOVA): {r2}")

    # Parse results
    final_results = {}
    for index in system_indexs:
        sys_conf = util_func.ALL_SYS[index]
        system = sys_conf[0]
        bench = sys_conf[1]
        alg = sys_conf[2] if len(sys_conf) == 3 else None
        output_name = f"{system}-{bench}-{alg}" if alg else f"{system}-{bench}"
        for rel in rel_bounds:
            # if (r2_results[rel][output_name] >= 0.9):
            #     final_results[output_name] = sample_ratios[rel][output_name]
            #     break
            print(f"{output_name},{r2_results[rel][output_name]:.4f},{ml_r2_results[rel][output_name]:.4f},"
                  f"{anova_r2_results[rel][output_name]:.4f},{sample_ratios[rel][output_name]:.4f}")

    # final_results = dict(sorted(final_results.items(), key=lambda item: item[1]))
    # for key, value in final_results.items():
    #     print(f"{key},{value:.4f}")


