import argparse

import pandas as pd
import numpy as np
import os

ALL_SYS = [("mysql", "tpcc"),
               ("aria", "tpcc", "Aria"),
               ("star", "ycsb"),
               ("cicada", "ycsb", "Cicada"),
               ("herd", "ycsb"),
               ("silo", "tpcc", "Silo"),
               ("calvin", "tpcc", "Calvin"),
               ("janus", "tpcc", "Janus"),
               ("gam", "tpcc"),
               ("drtm", "tpcc"),
               ("star", "tpcc"),
               ("tapir", "ycsb"),
               ("silo", "ycsb"),
               ("postgresql", "tpcc")]

GOOD_SYS = [("mysql", "tpcc"),
            ("aria","tpcc","Aria"),
            ("star","ycsb"),
            ("cicada","ycsb","Cicada"),
            ("herd","ycsb"),
            ("postgresql","tpcc")]

BAD_SYS = [("silo","tpcc","Silo"),
           ("calvin","tpcc","Calvin"),
           ("janus","tpcc","Janus"),
           ("gam","tpcc"),
           ("drtm","tpcc"),
           ("star","tpcc"),
           ("tapir","ycsb"),
           ("silo","ycsb")]

def get_raw_data(sys_conf):
    file_name = sys_conf[0]+"_"+sys_conf[1]
    data = pd.read_csv(f"csv/{file_name}.csv")
    return data

def get_factor_names(sys_conf):
    system = sys_conf[0]
    bench = sys_conf[1]
    if system == "star":
        if bench == "ycsb":
            para_names = ["partition_num", "threads", "zipf", "rw", "servers", "cross"]
        elif bench == "tpcc":
            para_names = ["partition_num", "threads", "servers", "cross"]
    elif system == "gam":
        para_names = ["warehouses", "threads", "cross", "servers"]
    elif system == "herd":
        para_names = ["num_keys", "value_size", "put_percent", "zipf"]
    elif system == "aria":
        para_names = ["WH", "Distributed", "threads"]
    elif system == "calvin":
        para_names = ["nnodes", "WH", "dist"]
    elif system == "cicada":
        if bench == "tpcc":
            para_names = ["threads", "warehouse_count"]
        if bench == "ycsb":
            para_names = ["threads", "record_size", "req_per_query", "total_count", "read_ratio", "zipf_theta"]
    elif system == "silo":
        if bench == "tpcc":
            para_names = ["threads", "warehouse", "remote"]
        if bench == "ycsb":
            para_names = ["threads", "keys", "ReadRatio"]
    elif system == "tapir":
        para_names = ["keys", "ReadRatio", "nShard", "nrep", "nclients"]
    elif system == "janus":
        para_names = ["clients", "shards", "threads", "replica"]
    elif system == "drtm":
        para_names = ["node", "threads", "scale_factor", "cross_warehouse"]
    elif system == "calvin2":
        para_names = ["WH", "#Distributed", "nodes"]
    elif system == "mysql":
        para_names = ["WH", "terminal", "num_chunk", "chunk_size"]
    elif system == "postgresql":
        para_names = ["terminal","shared_buffer","min_wal_size","max_wal_size","effective_cache_size"]

    return para_names

def save_r2_results(r2_results, output_dir, output_file):
    check_dir(output_dir)
    output = f"{output_dir}/{output_file}"
    index = [f"{r:.2f}" for r in np.arange(0.05, 1, 0.05)]
    df = pd.DataFrame(r2_results, columns=[str(i) for i in range(100)],
                      index=index)
    df = df.round(decimals=3)
    df.reset_index(inplace=True)
    df = df.rename(columns={'index': 'ratio'})
    df.to_csv(output, index=False)

def check_dir(dir_path):
    # Check if the directory exists
    if not os.path.exists(dir_path):
        # Create the directory
        os.makedirs(dir_path)
        print(f"Directory '{dir_path}' created.")
    else:
        print(f"Directory '{dir_path}' already exists.")


def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, choices=["random", "stratified", "balance", "dist-aware"],
                        help="sampling method: random | stratified | balance | dist-aware")
    parser.add_argument('--systems', nargs='+', type=int, help='index of systems to run')
    parser.add_argument('--all', type=bool, default=False, help='run all systems')

    return parser.parse_args()