import pandas as pd

# test wether two dfs have intersection.
# return true if no intersection else false
def test_split(df1, df2):
    tmp_pd = pd.merge(df1, df2, on=list(set(df1.columns)), how='inner')
    if len(tmp_pd) != 0:
        print("!!!ERROR on split data")
        exit(-1)