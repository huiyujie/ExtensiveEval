import pandas as pd
from itertools import product
import sys
sys.path.append("../")
from util_func import check_dir


def insert_missing_data(dataframe, param_names):
    dimensions = {}
    for term in param_names:
        dimensions[term] = dataframe[term].unique().tolist()
    full_factorial_df = pd.DataFrame(list(product(*dimensions.values())), columns=dimensions.keys())
    full_factorial_df = pd.merge(dataframe, full_factorial_df, how='outer', indicator=True)

    full_factorial_df.sort_values(by=param_names, inplace=True, ignore_index=True)
    full_factorial_df["tput"] = full_factorial_df["tput"].fillna(0)
    full_factorial_df["is_filled"] = full_factorial_df['tput'].apply(lambda x: True if x == 0 else False)

    dims = []
    for term in param_names:
        unique_values = full_factorial_df[term].unique().tolist()
        dims.append(len(unique_values))

    return full_factorial_df, dims

def save_tput(data, output_dir, output_name):
    check_dir(output_dir)
    data["tput"].to_csv(f"{output_dir}/{output_name}", index=False, header=False)



