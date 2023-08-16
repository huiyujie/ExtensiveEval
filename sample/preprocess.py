

def preprocess(system, bench, alg, input_data):
    data, factor_names = read_data(system, bench, alg, input_data)

    # filter out data with throughput <= 0
    data = data[data.iloc[:, -1] > 0]

    if system == "cicada":
        data["tput"] = data["tput"] * data["req_per_query"]

    # normalize throughput by the number of worker threads
    data["tput"] = data["tput"] / get_norm_numpy(system, data)

    return data, factor_names


def read_data(system, bench, alg, input_data):
    if system == "star":
        if bench == "ycsb":
            para_names = ["partition_num", "threads", "zipf", "rw", "servers", "cross"]
        elif bench == "tpcc":
            para_names = ["partition_num", "threads", "servers", "cross"]
    elif system == "gam":
        para_names = ["warehouses", "threads", "cross","servers"]
    elif system == "herd":
        para_names = ["num_keys", "value_size", "put_percent", "zipf"]
    elif system == "aria":
        algs = ["Aria", "AriaFB-1", "AriaFB-2", "AriaFB-4", "AriaFB-6", "Bohm",
                "Calvin-1", "Calvin-2", "Calvin-4", "Calvin-6", "Pwv"]
        if not alg or alg not in algs:
            print("need to specify an algorithm in " + str(algs))
            exit()
        para_names = ["WH", "Distributed", "threads"]
    elif system == "calvin":
        algs = ["2pl", "Calvin"]
        if not alg or alg not in algs:
            print("need to specify an algorithm in " + str(algs))
            exit()
        para_names= ["nnodes", "WH", "dist"]
    elif system == "cicada":
        algs = ["2PL", "Cicada", "ERMIA", "FOEDUS", "Hekaton", "MOCC", "Silo", "TicToc"]
        if not alg or alg not in algs:
            print("need to specify an algorithm in " + str(algs))
            exit()
        if bench == "tpcc":
            para_names = ["threads", "warehouse_count"]
        if bench == "ycsb":
            para_names = ["threads", "record_size", "req_per_query", "total_count", "read_ratio", "zipf_theta"]
    elif system == "silo":
        algs = ["PartitionedStore", "Silo"]
        if bench == "tpcc":
            if not alg or alg not in algs:
                print("need to specify an algorithm in " + str(algs))
                exit()
            para_names = ["threads","warehouse","remote"]
        if bench == "ycsb":
            para_names = ["threads", "keys", "ReadRatio"]
    elif system == "tapir":
        para_names = ["keys","ReadRatio","nShard","nrep","nclients"]
    elif system == "janus":
        para_names = ["clients","shards","threads","replica"]
    elif system == "drtm":
        para_names = ["node","threads","scale_factor","cross_warehouse"]
    elif system == "calvin2":
        para_names = ["WH","#Distributed","nodes"]
    elif system == "mysql":
        para_names = ["WH","terminal","num_chunk","chunk_size"]

    if alg:
        input_data = input_data[input_data['alg'] == alg]
    if system == "cicada" and bench == "tpcc":
        input_data = input_data[input_data['bench'] == "TPCC-FULL"]
    if system == "silo" and bench == "tpcc":
        input_data = input_data[input_data['trans'] == "new_order"]
    if system == "drtm":
        input_data = input_data.drop(columns=['tp_1', 'tp_2', 'tp_3'])

    return input_data, para_names


def get_norm_numpy(system, df):
    if system == "star" or system == "gam":
        return df["threads"].to_numpy() * df["servers"].to_numpy()
    elif system == "herd":
        return 1
    elif system == "drtm":
        return df["threads"].to_numpy() * df["node"].to_numpy()
    elif system == "aria" or system == "cicada" or system == "silo":
        return df["threads"].to_numpy()
    elif system == "calvin":
        return df["nnodes"].to_numpy()
    elif system == "tapir":
        return df["nrep"].to_numpy() * df["nShard"].to_numpy()
    elif system == "janus":
        return df["replica"].to_numpy() * df["shards"].to_numpy()
    elif system == "calvin2":
        return df["nodes"].to_numpy() * 12
    elif system == "mysql":
        return df["terminal"].to_numpy()
