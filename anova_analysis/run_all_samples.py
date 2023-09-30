import sys,os
path_to_sample_dir = 'ExtensiveEval/sample'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), path_to_sample_dir)))
import sample
import pandas as pd




def run_one_system(df, name, bench, seed=None, output_dir = "/ANOVA/all_samples"):
    
    # for method in ["random", "stratified", "balance", "dist-aware"]:
    for method in ["random", "dist-aware"]:
        if seed == None:
            samples, unsamples = sample.sample(df, method, name, bench, output_dir = output_dir)
        else:
            samples, unsamples = sample.sample(df, method, name, bench, output_dir = output_dir, seed = seed)
        


    
    

if __name__ == "__main__":
    

    data = pd.read_csv("/ANOVA/ExtensiveEval/csv/tapir_ycsb.csv")
    samples , unsamples = sample.sample(data, "dist-aware", "tapir", "ycsb", output_dir = None)