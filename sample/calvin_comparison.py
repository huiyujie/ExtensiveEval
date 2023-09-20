import pandas as pd

def read_calvin_data():
    original_data = pd.read_csv("./csv/calvin_tpcc.csv")
    revised_data = pd.read_csv("./csv/aria_tpcc.csv")

    original_data = original_data[original_data["nnodes"] == 1]
    original_data = original_data[original_data["alg"] == "Calvin"]
    original_data.drop(columns=["nnodes", "alg"], inplace=True)

    revised_data = revised_data[revised_data["alg"] == "Calvin-1"]
    revised_data = revised_data[revised_data["threads"] == 4]
    revised_data.drop(columns=["alg", "threads"], inplace=True)
    revised_data.rename(columns={"Distributed": "dist"}, inplace=True)
    conditions = (revised_data["WH"] == 256) | (revised_data["WH"] == 512) | (revised_data["WH"] == 1024)
    revised_data = revised_data[~conditions]

    return original_data, revised_data



if __name__ == "__main__":
    original, calvin_aria = read_calvin_data()