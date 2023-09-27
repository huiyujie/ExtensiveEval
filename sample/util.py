import pandas as pd
from sklearn.model_selection import train_test_split

# test wether two dfs have intersection.
# return true if no intersection else false
def test_split(df1, df2):
    tmp_pd = pd.merge(df1, df2, on=list(set(df1.columns)), how='inner')
    if len(tmp_pd) != 0:
        print("!!!ERROR on split data")
        exit(-1)

def split_sampleset(sample_data,random_state):
    train, test = train_test_split(sample_data,
                                    test_size=0.2,
                                    random_state=random_state,
                                    stratify=None)
    return train, test