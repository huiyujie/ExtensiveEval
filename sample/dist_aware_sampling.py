import numpy as np
import pandas as pd
import scipy as sp
import copy,os
from util import  split_sampleset


# ----------------------------find neighbors by distance--------------------------------------
def find_neighbor_var(unsample_idx,parms,neighbors_idx,clm_names):
    neighbors_df = parms.loc[parms['sample_idx'].isin(neighbors_idx[unsample_idx]) ,]
    pdist_condensed =  sp.spatial.distance.pdist(neighbors_df[clm_names])
    pdist = sp.spatial.distance.squareform(pdist_condensed)
    tput_np = neighbors_df['tput'].to_numpy()
    tput_pdiff = abs(np.subtract.outer(tput_np,tput_np))
    avg_var = np.divide(tput_pdiff,pdist)
    np.fill_diagonal(avg_var,0)
    return avg_var.sum(axis=0).mean()/2
# -------------------------------------------------------------------------------------------



def output_sample_result(output_file,ratio,random_seed,df_in,sample_idxs):
    unexp_idxs = list(set(df_in.index) - set(sample_idxs))
    exp = df_in.loc[sample_idxs].copy()
    unexp = df_in.loc[unexp_idxs].copy()
    
    exp_list = []
    unexp_list = []
    
    for r_split in range(10):
    
        exp_train, exp_test = split_sampleset(exp,r_split)

        if output_file:
            output_path_sample = f"{output_file}_sample@{ratio:.2f}_random@{random_seed}_split@{r_split}_exp.csv"
            output_path_unexp = f"{output_file}_sample@{ratio:.2f}_random@{random_seed}_split@{r_split}_unexp.csv"
            exp_train.to_csv(output_path_sample,index=False)
            exp_test.to_csv(output_path_unexp,index=False)

        exp_list.append(exp_train)
        unexp_list.append(exp_test)
    
    return exp_list,unexp_list
    
    
def sample_one_seed(df_in, output_file, random_seed):
    
    samples_list = [[0] * 19 for _ in range(10)]
    unsamples_list = [[0] * 19 for _ in range(10)]
    
    df_in = df_in.loc[df_in['tput'] > 0]
    df_in.reset_index(inplace=True,drop=True)
    columns = df_in.columns.to_list()
    if "trans" in columns:
        columns.remove("trans")
    if "alg" in columns:
        columns.remove("alg")
    columns_no_tput = copy.deepcopy(columns)
    columns_no_tput.remove("tput")
    
    parms = df_in[columns].copy()
    parms = (parms-parms.mean())/parms.std()
    parms['sampled'] = False
    full_size = parms.shape[0]
    step_size = full_size // 100 * 5
    rng = np.random.default_rng(random_seed)
    unsample_region = parms.loc[parms['sampled']==False]
    sample_ratio = 0.05

    while unsample_region.shape[0] > step_size:
        if sample_ratio == 0.05:
            newly_sampled = rng.choice(full_size,step_size,replace=False)
        else:
            # ----------------------------weighted random sampling--------------------------------------
            newly_sampled = rng.choice(unsample_region.index,step_size,replace=False,p=unsample_region['neighbors_var']/unsample_region['neighbors_var'].sum())
            # -------------------------------------------------------------------------------------------
            
            # ---------------------------------greedy sampling-------------------------------------------
            # newly_sampled = unsample_region.sort_values(by="neighbors_var",ascending=False).iloc[:step_size].index

        parms.loc[newly_sampled,'sampled'] = True
        
        exp,unexp = output_sample_result(output_file,sample_ratio,random_seed,df_in,parms.loc[parms['sampled']==True].index)
        
        for r_split in range(10):
            samples_list[r_split][round(sample_ratio/0.05)-1] = exp[r_split]
            unsamples_list[r_split][round(sample_ratio/0.05)-1] = unexp[r_split]
            
        
        if abs(sample_ratio - 0.95) < 1e-3:
            break
        sampled_size = parms.loc[parms['sampled']==True].shape[0]
        unsampled_size = full_size - sampled_size
        # print(f"sampled size: {sampled_size}, unsampled size: {unsampled_size}")
        # count sampled and unsampled points separately
        parms['sample_idx'] = -1
        parms['unsample_idx'] = -1
        parms.loc[parms['sampled']==True,'sample_idx'] = np.arange(sampled_size,dtype=int)
        parms.loc[parms['sampled']==False,'unsample_idx'] = np.arange(unsampled_size,dtype=int)
        # data to build kd-tree
        sampled_x = parms.loc[parms['sampled']==True,columns_no_tput]
        # data to query
        unsampled_x = parms.loc[parms['sampled']==False,columns_no_tput]
        # ----------------------------find neighbors by distance--------------------------------------
        kdtree = sp.spatial.KDTree(sampled_x)
        # print(f"tree size: {kdtree.data.shape[0]}")
        # query result
        _,neighbors_idx = kdtree.query(unsampled_x,k=2**len(columns_no_tput))
        # ----------------------------find neighbors by distance--------------------------------------
        # gather neighbors' variance
        parms.loc[parms['sampled']==False,'neighbors_var'] = parms.loc[parms['sampled']==False,'unsample_idx'].apply(find_neighbor_var,parms=parms,neighbors_idx=neighbors_idx,clm_names=columns_no_tput)
        unsample_region = parms.loc[parms['sampled']==False]
        
        sample_ratio += 0.05
    
    return samples_list, unsamples_list
    
    
def dist_aware_sampling(data, output_file = None, seed = None):
    
    num_seeds = 100
    
    if seed is not None:
        exp,unexp = sample_one_seed(data, output_file, seed)
        return [exp], [unexp]
    else:
        samples_list = []
        unsamples_list = []
        for i in range(num_seeds):
            exp,unexp = sample_one_seed(data, output_file, i)
            samples_list.append(exp)
            unsamples_list.append(unexp)
            

        return samples_list, unsamples_list