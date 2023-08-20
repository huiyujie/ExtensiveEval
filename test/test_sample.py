import sys,os
path_to_sample_dir = '../sample'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), path_to_sample_dir)))
import sample
import pandas as pd

if __name__ == "__main__":
    

    data = pd.read_csv("/ANOVA/ExtensiveEval/csv/tapir_ycsb.csv")
    samples = sample.sample(data, "balance", "tapir", "ycsb", seed=0)