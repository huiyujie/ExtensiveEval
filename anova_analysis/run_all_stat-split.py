import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr,linregress
from itertools import combinations
import re
import matplotlib.pyplot as plt
import warnings
import os,shutil,json,glob,sys
import numpy as np
from sklearn.model_selection import train_test_split
from similaritymeasures import frechet_dist
from kneed import KneeLocator
import  scipy.signal.signaltools
import time
import multiprocessing as mp
from sklearn.metrics import r2_score

path_to_sample_dir = '../sample'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), path_to_sample_dir)))
import sample
from preprocess import preprocess
from random_sampling import random_sampling


def _centered(arr, newsize):
    # Return the center newsize portion of the array.
    newsize = np.asarray(newsize)
    currsize = np.array(arr.shape)
    startind = (currsize - newsize) // 2
    endind = startind + newsize
    myslice = [slice(startind[k], endind[k]) for k in range(len(endind))]
    return arr[tuple(myslice)]

scipy.signal.signaltools._centered = _centered


import statsmodels.api as sm
from statsmodels.formula.api import ols

warnings.filterwarnings("ignore")


PLOT=False
PLOT_turning=False
SAVE_INTERIM_FILE=False
# PLOT=True
# PLOT_turning=True
COMPARE_FILE="COMPARE.LOG"
plt.rc('font', size=24)
plt.rc('axes', labelsize=20)
plt.rc('axes',titlesize=24)   

def init_terms(terms):
    initial_term = []
    initial_term.append(list(combinations(terms, 1)))
    initial_term.append(list(combinations(terms, 2)))
    new_terms = []
    for term in initial_term[0]:
        new_terms.append("C(" + term[0] + ")")
    for term in initial_term[1]:
        new_terms.append("C(" + term[0] + "):C(" + term[1] + ")")
    return new_terms

def init_terms_single(terms):
    initial_term = []
    initial_term.append(list(combinations(terms, 1)))
    new_terms = []
    for term in initial_term[0]:
        new_terms.append("C(" + term[0] + ")")
    return new_terms

def format_equation(terms):
    equation = "tput~"

    for term in terms:
        equation += term + "+"
    equation = equation[:-1]
    return equation


def remove_terms_single(params, df):
    removed_terms = {}
    terms = init_terms_single(params)
    remove = ["-1"]
    threshold = 0.05
    while len(remove) != 0:
        eq = format_equation(terms)
        if len(terms) < 1:
            eq = format_equation(init_terms_single(params))
            model = ols(eq, data=df).fit()
            return model,None,True,None
        model = ols(eq, data=df).fit()
        anova_table = sm.stats.anova_lm(model, typ=3)
        anova_table = anova_table[~anova_table.index.isin(["Intercept"])]
        invalid = list(anova_table[anova_table["PR(>F)"] > threshold].index)

        if len(invalid) > 0:
            remove = anova_table["PR(>F)"].idxmax()
            terms.remove(remove)
            removed_terms[remove] = anova_table.loc[remove,'PR(>F)']
        else:
            remove = []
    
    return model,removed_terms,False,1
    
def remove_terms_combine(params, df):
    removed_terms = {}
    terms = init_terms(params)
    remove = ["-1"]
    threshold = 0.05
    
    while len(remove) != 0:
        eq = format_equation(terms)
        if len(terms) < 1:
            eq = format_equation(init_terms(params))
            model = ols(eq, data=df).fit()
            return model,None,True,None
        model = ols(eq, data=df).fit()
        anova_table = sm.stats.anova_lm(model, typ=3)
        anova_table = anova_table[~anova_table.index.isin(["Intercept"])]
        invalid = list(anova_table[anova_table["PR(>F)"] > threshold].index)
        

        if len(invalid) > 0:
            remove = anova_table["PR(>F)"].idxmax()
            terms.remove(remove)
            removed_terms[remove] = anova_table.loc[remove,'PR(>F)']
        else:
            remove = []
        
    return model,removed_terms,False,2

def remove_terms(params,df):
    try:
        return remove_terms_combine(params,df)
    except ValueError:
        return remove_terms_single(params,df)


def check_consistent(model,terms):
    cof_table = model.summary2().tables[1]['Coef.'].drop(['Intercept'])
    cof_table = cof_table.reset_index()
    parms_in_model =  cof_table['index'].apply(lambda x:re.sub("\[(T\.)?[^:]+\]","",x)).unique()
    for parm in parms_in_model:
        if parm not in terms:
            return False
    return True


def detect_knee(df,column,spear):
    if spear < 0:
        knd= KneeLocator(x=df[column],y=df['Coef.'],curve="convex",direction="decreasing",online=True)
    else:
        knd = KneeLocator(x=df[column],y=df['Coef.'],curve="concave",direction="increasing",online=True)
    
    if not knd.knee:
        return False,False
    else:
        div_point = knd.knee
        df_low = df.loc[df[column]<=div_point,:]
        df_high = df.loc[df[column]>=div_point,:]
        m1 = linregress(df_low[column],df_low['Coef.']).slope
        m2 = linregress(df_high[column],df_high['Coef.']).slope
        turning_ratio = abs((m2-m1)/m1)
        
        return div_point,turning_ratio
        
def detect_turning_coef(df,column,thd=0.8):    
    col_data = df[column].unique()
    col_data.sort()
    
    spear_overall = spearmanr(df[column],df['Coef.']).correlation
    
    if len(col_data)<3:
        return False,(spear_overall,-999),False
    
    if abs(spear_overall) >= 0.98:
        div_point,turning_ratio = detect_knee(df,column,spear_overall)
        return div_point,(spear_overall,-888),turning_ratio
        
    max_diff = -888
    div = -999
    max_spear1=-888
    max_spear2=-888
    turning_ratio = -999

    for div_point in col_data[1:-1]:
        df_low = df.loc[df[column]<=div_point,:]
        spear_low = spearmanr(df_low[column],df_low['Coef.']).correlation
        df_high = df.loc[df[column]>=div_point,:]
        spear_high = spearmanr(df_high[column],df_high['Coef.']).correlation
        diff = abs(spear_low-spear_high)
        
        
        if diff > max_diff:
            max_diff = diff
            div = div_point
            max_spear1 = spear_low
            max_spear2 = spear_high
            m1 = linregress(df_low[column],df_low['Coef.']).slope
            m2 = linregress(df_high[column],df_high['Coef.']).slope
            turning_ratio = abs((m2-m1)/m1)
    
    
    if abs(max_diff) > thd and max_diff!=-888:
        return div, (max_spear1 , max_spear2),turning_ratio

    return False,(spear_overall,-999),False

def spear_message_one(spear):
    if spear < -0.5:
        message = "decreasing"
    elif spear < 0.5:
        message = "flat"
    else:
        message = "increasing"
    return message

def spear_message(spear1,spear2):
    msg1 = spear_message_one(spear1)
    msg2 = spear_message_one(spear2)
    if msg1 == msg2:
        return f"This is a knee point on a {msg1} curve\n"
    else:
        return f"Trend: First {msg1}, Then {msg2}\n"

def find_in_anova(df,x,key):
    try:
        res = df.loc[x,key]
        return res
    except KeyError:
        return pd.NA
    
def ANOVA_coef_helper(df,name,DIR_NAME,parms, model,removed_terms,fail_to_remove,sampled,unsampled_df):
    
    
    if SAVE_INTERIM_FILE:
        open(f"{DIR_NAME}/{name}.txt",'w').close()
        f = open(f"{DIR_NAME}/{name}.txt",'a')
    else:
        f = open(os.devnull,'w')
    
    print("ANOVA_analysis",file=f)
    
    if not removed_terms:
        df_removed_terms = pd.DataFrame.from_records([], columns=['term','p-value'])
    else:
        df_removed_terms = pd.DataFrame.from_records(list(removed_terms.items()), columns=['term','p-value'])
    
    if not fail_to_remove:
        if removed_terms:
            print("Insignificant terms removed: ",file=f)
            print(removed_terms,file=f)
            
        else:
            print("All terms are significant",file=f)
    else:
        print("Error: failed to remove insignificant terms, keeping all terms",file=f)
    
    print("\n\n",file=f)
    cof_table = model.summary2().tables[1]['Coef.'].drop(['Intercept'])
    cof_table = cof_table.reset_index()
    cof_table['parms'] = cof_table['index'].apply(lambda x:re.sub("\[(T\.)?[^:]+\]","",x))
    anova_table = sm.stats.anova_lm(model, typ=3)
    anova_table['per_var_explained'] = anova_table['sum_sq']/anova_table['sum_sq'].sum()
    anova_table = anova_table.drop(['Residual'])
    anova_table.sort_values(['PR(>F)','per_var_explained'],inplace=True,ascending=[True,False])
    assert   anova_table['per_var_explained'].sum() - model.rsquared < 1e5, "R-square and per_var_explained not matched"
    
    with open(f"{DIR_NAME}/{name}.r2","w") as r2f:
        print(f"{model.rsquared}",file=r2f)
        if unsampled_df is not None:
            try:
                print(f"{r2_score(unsampled_df['tput'],model.predict(unsampled_df))}",file=r2f)
            except:
                pass
    
    print("--------------------------------------------------------",file=f)
    print(f"Model r-squared: {model.rsquared:.4f}",file=f)
    print("--------------------------------------------------------\n",file=f)
    print("ANOVA Terms with at leat 10% variance explained:\n",file=f)
    print(anova_table.loc[anova_table['per_var_explained']>0.1,['PR(>F)','per_var_explained']].to_string(),file=f)
    print("--------------------------------------------------------",file=f)
    
    # cof_table['PR>F'] = cof_table['parms'].apply(lambda x:anova_table.loc[x,'PR(>F)'])
    cof_table['PR>F'] = cof_table['parms'].apply(lambda x:find_in_anova(anova_table,x,'PR(>F)'))
    # cof_table['per_var_explained'] = cof_table['parms'].apply(lambda x:anova_table.loc[x,'per_var_explained'])
    cof_table['per_var_explained'] = cof_table['parms'].apply(lambda x:find_in_anova(anova_table,x,'per_var_explained'))
    cof_table.sort_values(['PR>F','per_var_explained','parms'],ascending=[True,False,True],inplace=True)
    # print("Significant terms:",file=f)
    # print(cof_table['parms'].unique(),file=f)
    # print("\n\n",file=f)
    print("Significant terms and turning point:",file=f)
    print("--------------------------------------------------------",file=f)
    cof_table_group = cof_table.groupby(["PR>F","parms","per_var_explained"],dropna=False)
    num_significant_parms = cof_table_group.ngroups
    if PLOT:
        fig,axes = plt.subplots(num_significant_parms,1, constrained_layout=True,figsize=(16,10*num_significant_parms),squeeze=False)
    
    idx_parm_group = -1
    trend_data_list = []
    cof_data_list = []
    parms_list = cof_table.parms.unique()
    
    
    # for (PRf,parm_orig,per_var_explained),group in cof_table_group:
    for parm_orig in parms_list:
        
        idx_parm_group += 1
        
        group = cof_table[cof_table.parms==parm_orig].copy()
        PRf = group["PR>F"].iloc[0]
        per_var_explained = group["per_var_explained"].iloc[0]
        
        if pd.isna(per_var_explained):
            continue
        
        if per_var_explained < 0.1:
            df_removed_terms = pd.concat([df_removed_terms,pd.DataFrame.from_records([{'term':parm_orig,'p-value':PRf,'per_var_explained':per_var_explained}])],ignore_index=True)
            # df_removed_terms = df_removed_terms.append({'term':parm_orig,'p-value':PRf,'per_var_explained':per_var_explained},ignore_index=True)
            continue
        
        parm = parm_orig.replace("C(","").replace(")","")
        print(f"\nTerm: {parm}, Percentage of variance explained:{per_var_explained:0.4f}",file=f)
        if ":" in parm:
            parm_0 = parm.split(":")[0]
            parm_1 = parm.split(":")[1]
            
            
            ## TODO: FIX T.
            group[parm_0] = group['index'].apply(lambda x:x.split(':')[0])
            group[parm_1] = group['index'].apply(lambda x:x.split(':')[1])
            group[parm_0] = group[parm_0].apply(lambda x:re.sub("C\(\w+\)","",x.replace("[","").replace("]","").replace("T.","")))
            group[parm_1] = group[parm_1].apply(lambda x:re.sub("C\(\w+\)","",x.replace("[","").replace("]","").replace("T.","")))            
            
            group[parm_0] = group[parm_0].astype('float')
            group[parm_1] = group[parm_1].astype('float')
            
            # group[parm_0] = group['index'].apply(lambda x:float(x.split(':')[0].replace(")[","").replace("]","").split("T.")[1]))
            # group[parm_1] = group['index'].apply(lambda x:float(x.split(':')[1].replace(")[","").replace("]","").split("T.")[1]))
            if PLOT:
                sns.heatmap(group.pivot(index=parm_0,columns=parm_1,values='Coef.'),ax=axes[idx_parm_group,0])
                axes[idx_parm_group,0].set_title(f"{name}, Coef of {parm}, per_var:{per_var_explained:.4f}")
            
            spear_group = []
            
            all_turning = True
            cof_data = {"#parm":2,'term_raw':parm_orig}
            cof_data['parm1'] = parm_0
            cof_data['parm2'] = parm_1
            cof_data['parm1_val'] = group[parm_0].to_list()
            cof_data['parm2_val'] = group[parm_1].to_list()
            cof_data['Coef'] = group['Coef.'].to_list()
            num_of_entries_in_group = len(cof_data['parm1_val'])
            
            for idx1,group1 in group.groupby(parm_0):
                
                
                turning_coef,(spear1,spear2),turning_ratio = detect_turning_coef(group1,parm_1)
                spear_row = spearmanr(group1[parm_1],group1['Coef.']).correlation
                spear_group.append(spear_row)
                
                trend_data = {'term_raw':parm_orig}
                trend_data['PR(>F)'] = PRf
                trend_data['per_var_explained'] = per_var_explained
                trend_data['#parm'] = 2
                
                trend_data['fixed_parm_name'] = parm_0
                trend_data['var_parm_name'] = parm_1
                trend_data['fixed_parm_val'] = idx1
                trend_data['spear_all'] = spear_row
                trend_data['weight'] = num_of_entries_in_group
                
                if turning_coef:
                    all_turning &= True
                    dff = df.loc[df[trend_data['fixed_parm_name']]==trend_data['fixed_parm_val']]
                    settings = parms.copy()
                    settings.remove(trend_data['var_parm_name'])
                    if dff.groupby(by=settings).ngroups > 1:
                        trend_data['turning'] = turning_coef
                        trend_data['turning_spear1'] = spear1
                        trend_data['turning_spear2'] = spear2
                        trend_data["turning_ratio"] = turning_ratio
                        print(f"Turning point: {parm_1}={turning_coef},{parm_0} fixed at{idx1}",file=f)
                        print(spear_message(spear1,spear2),file=f)
                else:
                     all_turning &= False
                     print(f"When fixing {parm_0}@{idx1}, Overall trend: {spear_message_one(spear1)} with spearman coef = {spear1:.2f}\n",file=f)
                
                trend_data_list.append(trend_data)
            
            if all([x > 0.5 for x in spear_group]):
                print(f"**Overall, Coef. increase with {parm_1} while {parm_0} is fixed\n",file=f)
                
            if all([x < -0.5 for x in spear_group]):
                print(f"**Overall, Coef. decrease with {parm_1} while {parm_0} is fixed\n",file=f)
                
            if all([x > -0.2 for x in spear_group]) and all([x < 0.2 for x in spear_group]):
                print(f"**Overall, Coef. flat while {parm_0} is fixed\n",file=f)
                
            if all_turning:
                print(f"**Overall, Coef. all have a turning point while {parm_0} is fixed\n",file=f)
            
            spear_group = []
            all_turning = True
            for idx2,group2 in group.groupby(parm_1):
                turning_coef,(spear1,spear2), turning_ratio= detect_turning_coef(group2,parm_0)
                spear_row = spearmanr(group2[parm_0],group2['Coef.']).correlation
                spear_group.append(spear_row)
                
                trend_data = {'term_raw':parm_orig}
                trend_data['PR(>F)'] = PRf
                trend_data['per_var_explained'] = per_var_explained
                trend_data['#parm'] = 2
                
                trend_data['fixed_parm_name'] = parm_1
                trend_data['var_parm_name'] = parm_0
                trend_data['fixed_parm_val'] = idx2
                trend_data['spear_all'] = spear_row
                trend_data['weight'] = num_of_entries_in_group
                
                if turning_coef:
                    all_turning &= True
                    dff = df.loc[df[trend_data['fixed_parm_name']]==trend_data['fixed_parm_val']]
                    settings = parms.copy()
                    settings.remove(trend_data['var_parm_name'])
                    if dff.groupby(by=settings).ngroups > 1:
                        trend_data['turning'] = turning_coef
                        trend_data['turning_spear1'] = spear1
                        trend_data['turning_spear2'] = spear2
                        trend_data["turning_ratio"] = turning_ratio
                        print(f"Turning point: {parm_0}={turning_coef},{parm_1} fixed at {idx2}",file=f)
                        print(spear_message(spear1,spear2),file=f)
                else:
                    all_turning &= False
                    print(f"When fixing {parm_1}@{idx2}, Overall trend: {spear_message_one(spear1)} with spearman coef = {spear1:.2f}\n",file=f)

                trend_data_list.append(trend_data)
                
            if all([x > 0.5 for x in spear_group]):
                print(f"**Overall, Coef. increase with {parm_0} while {parm_1} is fixed\n",file=f)
                
            if all([x < -0.5 for x in spear_group]):
                print(f"**Overall, Coef. decrease with {parm_0} while {parm_1} is fixed\n",file=f)
                
            if all([x > -0.2 for x in spear_group]) and all([x < 0.2 for x in spear_group]):
                print(f"**Overall, Coef. flat while {parm_1} is fixed\n",file=f)

            if all_turning:
                print(f"**Overall, Coef. all have a turning point while {parm_0} is fixed\n",file=f)            
            # turning_coef_pairs = list(set(turning_coef_pairs))
            # for (coe1,coe2) in turning_coef_pairs:
                # print(f"Turning point:{parm_0}={coe1},{parm_1}={coe2}",file=f)
        else:            
            cof_data = {"#parm":1,'term_raw':parm_orig}
            cof_data['parm0'] = parm
            cof_data['Coef'] = group['Coef.'].to_list()
            
            group['index_value'] = group['index'].apply(lambda x:float(x.replace(")[","").replace("]","").split("T.")[1]))
            cof_data['parm0_val'] = group['index_value'].to_list()
            
            if PLOT:
                axes[idx_parm_group,0].plot(group['index_value'],group['Coef.'],"o--")
                axes[idx_parm_group,0].set_title(f"{name}, Coef. of {parm}, per_var:{per_var_explained:.4f}")
            
            turning_coef,(spear1,spear2),turning_ratio = detect_turning_coef(group,"index_value")
            
            trend_data = {'term_raw':parm_orig}
            trend_data['PR(>F)'] = PRf
            trend_data['per_var_explained'] = per_var_explained
            
            trend_data['#parm'] = 1
            trend_data['parm_name'] = parm
            trend_data['spear_all'] = spearmanr(group['index_value'],group['Coef.']).correlation 
            trend_data['weight'] = 1
            
            if turning_coef:
                settings = parms.copy()
                settings.remove(trend_data['parm_name'])
                if df.groupby(by=settings).ngroups > 1:
                    trend_data['turning'] = turning_coef
                    trend_data['turning_spear1'] = spear1
                    trend_data['turning_spear2'] = spear2
                    trend_data['turning_ratio'] = turning_ratio
                    print(f"Turning point: {parm}={turning_coef}",file=f)
                    print(spear_message(spear1,spear2),file=f)
                    if PLOT:
                        axes[idx_parm_group,0].axvline(x=turning_coef,color='r',linewidth=3)
            else:
                print(f"Overall trend: {spear_message_one(spear1)} with spearman coef = {spear1:.2f}",file=f)

            
            trend_data_list.append(trend_data)
        
        cof_data_list.append(cof_data)
        print("-------------------------------------------------",file=f)
        
    print('\n\n',file=f)
    print("==================================================================================",file=f)
    print("Full ANOVA coef table",file=f)
    
    print(cof_table.to_string(index=False),file=f)
    
    if SAVE_INTERIM_FILE:
        df_removed_terms.to_csv(f"{DIR_NAME}/{name}___removed_terms.csv",index=False)
    
    if (len(trend_data_list) == 0):
        f.close()
        open(f"{DIR_NAME}/{name}___trend.csv","w").close()
        open(f"{DIR_NAME}/{name}___cof.csv","w").close()
        return
        

    df_trend = pd.DataFrame.from_records(trend_data_list)
    
    try:
        df_trend['delta_spear'] = df_trend['turning_spear1'] - df_trend['turning_spear2']
        df_trend['delta_spear'] = df_trend['delta_spear'].abs()
    except KeyError:
        df_trend['delta_spear'] = pd.NA
        
    if 'turning_ratio' not in df_trend.columns:
        df_trend['combined_rank'] = df_trend['per_var_explained']
    else:
        df_trend['combined_rank'] = df_trend['per_var_explained'] * df_trend['turning_ratio']
    # df_trend['combined_rank'] = np.sqrt(df_trend['per_var_explained'].rank(method='max',ascending=False)*df_trend['delta_spear'].rank(method='max',ascending=False))
    

    
    df_trend.sort_values(['combined_rank','per_var_explained','delta_spear'],inplace=True,ascending=[False,False,False])
    column_order = ["term_raw","PR(>F)","per_var_explained","#parm","parm_name","spear_all","turning","turning_spear1","turning_spear2","turning_ratio","fixed_parm_name","var_parm_name","fixed_parm_val","delta_spear","combined_rank","weight"]
    for col in column_order:
        if col not in df_trend.columns:
            df_trend[col] = pd.NA
    
    df_trend = df_trend[column_order]
    df_trend.to_csv(f"{DIR_NAME}/{name}___trend.csv",index=False)
    if not sampled:
        print(f"{name},{len(init_terms(parms))},{len(list(df_trend.term_raw.unique()))},{df_trend.loc[~df_trend['turning'].isna()].shape[0]}")

    df_cof = pd.DataFrame.from_records(cof_data_list)
    column_order = ["#parm","term_raw","parm1","parm2","parm1_val","parm2_val","Coef","parm0","parm0_val"]
    for col in column_order:
        if col not in df_cof.columns:
            df_cof[col] = pd.NA
    df_cof = df_cof[column_order]
    df_cof.to_csv(f"{DIR_NAME}/{name}___cof.csv",index=False)
    
    
    
    if PLOT:
        fig.savefig(f"{DIR_NAME}/{name}.png",facecolor='white')
    f.close()    
    
def ANOVA_coef(df,name,DIR_NAME,parms_to_exclude=['alg','tput'],sampled=False,unsampled_df = None):

    
    parms = df.columns.to_list()
    if name == "silo-Silo_tpcc" or name.startswith('silo-Silo_tpcc'):
        parms_to_exclude.append("trans")
    for parm_exc in parms_to_exclude:
        if parm_exc not in parms:
            continue
        parms.remove(parm_exc)

    if sampled:
        singles_pattern = ["aria-Aria_tpcc","calvin-Calvin_tpcc","herd_ycsb","aria-Calvin-comp_tpcc","calvin-Calvin-comp_tpcc"]
        if any([name.startswith(sp) for sp in singles_pattern]):
            model,removed_terms,fail_to_remove,stop_at = remove_terms_single(parms,df)
        else:
            model,removed_terms,fail_to_remove,stop_at = remove_terms_combine(parms,df)
        
        with open(f"{DIR_NAME}/{name}_stopped_at",'w') as f:
            print(stop_at,file=f)
        
        ANOVA_coef_helper(df,name,DIR_NAME,parms, model,removed_terms,fail_to_remove,sampled,unsampled_df)
    else:
        try:
            DIR_NAME_SINGLE = DIR_NAME + "_SINGLE"
            if not os.path.exists(DIR_NAME_SINGLE):
                os.makedirs(DIR_NAME_SINGLE)
            model,removed_terms,fail_to_remove,stop_at = remove_terms_single(parms,df)
            ANOVA_coef_helper(df,name,DIR_NAME_SINGLE,parms, model,removed_terms,fail_to_remove,sampled,unsampled_df)
        except ValueError as e:
            print(f"Could not conduct analysis on full data set with single terms for {name}")
            # raise e
        
        try:
            DIR_NAME_COMBINE = DIR_NAME + "_COMBINE"
            if not os.path.exists(DIR_NAME_COMBINE):
                os.makedirs(DIR_NAME_COMBINE)
            model,removed_terms,fail_to_remove,stop_at = remove_terms_combine(parms,df)
            ANOVA_coef_helper(df,name,DIR_NAME_COMBINE,parms, model,removed_terms,fail_to_remove,sampled,unsampled_df)
        except ValueError as e:
            print(f"Could not conduct analysis on full data set with combined terms for {name}")
            # raise e
        
def plot_turning_point(df,name,DIR_NAME,parms_to_exclude=['alg','tput']):
    
    
    trend_df = pd.read_csv(f"{DIR_NAME}/{name}___trend.csv")
    
    trend_df = trend_df[trend_df.per_var_explained > 0.1]
    
    #consolidate turning points

    try:
        consolidate_candidate = trend_df.loc[(trend_df['#parm']==2) & ~(trend_df['turning'].isna())]
    except KeyError:
        return
    
    idx_to_remove = []

    for idx,row in consolidate_candidate.iterrows():
        if idx in idx_to_remove:
            continue
        cond = consolidate_candidate['term_raw'] == row['term_raw']
        cond &= consolidate_candidate['fixed_parm_val']==row['turning']
        cond &= consolidate_candidate['turning']==row['fixed_parm_val']
        if  any(cond):
            idx_to_remove.append(consolidate_candidate[cond].index.values[0])
    
    turning_df = trend_df.loc[~(trend_df['turning'].isna())].copy()
    turning_df = turning_df.drop(index=idx_to_remove)
    turning_df = turning_df.sort_values(by=['combined_rank'],ascending=False).reset_index()
    
    
    num_turning_point = min(5,turning_df.shape[0])
    parms = df.columns.to_list()
    for parm_exc in parms_to_exclude:
        if parm_exc not in parms:
            continue
        parms.remove(parm_exc)


    if num_turning_point == 0:
        return
    
    fig,axes = plt.subplots(num_turning_point,2, constrained_layout=True,figsize=(32,10*num_turning_point),squeeze=False)
  
    cof_df = pd.read_csv(f"{DIR_NAME}/{name}___cof.csv")

    for idx in range(num_turning_point):
        
        row = turning_df.iloc[idx,:]
        setting = parms.copy()

        
        if row['#parm'] == 1:
            message = f"Terms={row['term_raw']}\nPercent_Var_Explained={row['per_var_explained']:.4f}\nturning_ratio={row['turning_ratio']:.2f}"
            setting.remove(row['parm_name'])
            for _,gp in df.groupby(by=setting):
                axes[idx,0].plot(gp[row['parm_name']],gp['tput'],'o--')

            axes[idx,0].axvline(x=row['turning'],color='r',linewidth=3)
            axes[idx,0].set_xlabel(row['parm_name'])
            axes[idx,0].set_ylabel("tput")
            
            coef = json.loads(cof_df.loc[cof_df['term_raw'] == row['term_raw']]['Coef'].iloc[0])
            parm = json.loads(cof_df.loc[cof_df['term_raw'] == row['term_raw']]['parm0_val'].iloc[0])
            
            axes[idx,1].plot(parm,coef,"o-")
            axes[idx,1].axvline(x=row['turning'],color='r',linewidth=3)
            axes[idx,1].set_xlabel(row['parm_name'])
            axes[idx,1].set_ylabel("Coef.") 
            axes[idx,1].annotate(message, xy=(0.6, 0.05), xycoords='axes fraction')
        
        else:
            message = f"Terms={row['term_raw']}\n{row['fixed_parm_name']} fixed at {row['fixed_parm_val']}\nPercent_Var_Explained={row['per_var_explained']:.4f}\nturning_ratio={row['turning_ratio']:.2f}"
            setting.remove(row['var_parm_name'])
            dff = df.loc[df[row['fixed_parm_name']]==row['fixed_parm_val']]
            for _,gp in dff.groupby(by=setting):
                axes[idx,0].plot(gp[row['var_parm_name']],gp['tput'],'o--')
            
            axes[idx,0].axvline(x=row['turning'],color='r',linewidth=3)
            axes[idx,0].set_xlabel(row['var_parm_name'])
            axes[idx,0].set_ylabel("tput")
            
            term_df = cof_df.loc[cof_df['term_raw'] == row['term_raw']]
            coef = json.loads(term_df['Coef'].iloc[0])
            parm1 = json.loads(term_df['parm1_val'].iloc[0])
            parm2 = json.loads(term_df['parm2_val'].iloc[0])
            
            coef = pd.Series(coef)
            parm1 = pd.Series(parm1)
            parm2 = pd.Series(parm2)      

            if row['fixed_parm_name'] == term_df['parm1'].iloc[0]:
                idx_coe = parm1 == row['fixed_parm_val']
                axes[idx,1].plot(parm2[idx_coe],coef[idx_coe],"o-")
            else:
                idx_coe = parm2 == row['fixed_parm_val']
                axes[idx,1].plot(parm1[idx_coe],coef[idx_coe],"o-")
            
            axes[idx,1].axvline(x=row['turning'],color='r',linewidth=3)
            axes[idx,1].set_xlabel(row['var_parm_name'])
            axes[idx,1].set_ylabel("Coef.") 
            axes[idx,1].annotate(message, xy=(0.6, 0.05), xycoords='axes fraction')

        


    fig.savefig(f"{DIR_NAME}/{name}_turning_point.png",facecolor='white')    


    

# def stat_for_single_alg(df_in,name,result_dir=None):
def stat_for_single_alg(args_in):
    
    unsampled_df = None
    
    if len(args_in) == 2:
        df_in,name = args_in
        result_dir = None
    elif len(args_in) == 3:
        df_in,name,result_dir = args_in
    elif len(args_in) == 4:
        df_in,name,result_dir,unsampled_df = args_in
    else:
        raise ValueError("Wrong number of arguments")
    
    parms_to_exclude=['alg','tput']
    if isinstance(df_in,bool):
        return
    df = df_in.copy()
    try:
        if not result_dir:
                DIR_NAME = "/ANOVA/ANOVA_result_full"
                # df.to_csv(f"csv_full_interm/{name}.csv",index=False)
        else:
                DIR_NAME = result_dir
        
        if not os.path.exists(DIR_NAME):
            os.makedirs(DIR_NAME)
        
        
        if result_dir:
            ANOVA_coef(df,name,DIR_NAME,parms_to_exclude=parms_to_exclude,sampled=True,unsampled_df=unsampled_df)
        else:
            ANOVA_coef(df,name,DIR_NAME,parms_to_exclude=parms_to_exclude,unsampled_df=unsampled_df)
            
        if PLOT_turning:
            plot_turning_point(df,name,DIR_NAME,parms_to_exclude=parms_to_exclude)
    except:
        # traceback.print_exc()
        raise
        
# def compare_dropped_terms(full_csv,sample_csv,out_path):
#     try:
#         df1 = pd.read_csv(full_csv).set_index('term')
#         df2 = pd.read_csv(sample_csv).set_index('term')
#     except FileNotFoundError:
#         return
#     union = pd.concat([df1,df2],axis=0)
#     diff_df = union.drop(df1.index.intersection(df2.index))
#     if diff_df.shape[0] > 1:
#         diff_df.to_csv(out_path)


# def compare_turning_diff(full_csv,sample_csv,out_path):
#     if os.path.exists(out_path):
#         os.remove(out_path)
    
#     try:
#         df1 = pd.read_csv(full_csv).set_index(['term_raw','fixed_parm_name','fixed_parm_val'])
#         df2 = pd.read_csv(sample_csv).set_index(['term_raw','fixed_parm_name','fixed_parm_val'])
#     except (FileNotFoundError, pd.errors.EmptyDataError):
#         return
#     df1 = df1.loc[(~df1.turning.isna()) & (df1.turning_spear2 != -888),'delta_spear']
#     df2 = df2.loc[(~df2.turning.isna()) & (df2.turning_spear2 != -888),'delta_spear']
#     union = pd.concat([df1,df2],axis=0)
#     diff_df = union.drop(df1.index.intersection(df2.index))
#     if diff_df.shape[0] > 1:
#         diff_df.to_csv(out_path)

def compare_trend(full_trend_csv,sample_trend_csv,exp_name,out_dir,empty_d2 = False, stratified_term = False):
    if not stratified_term:
        out_path = os.path.join(out_dir,f"{exp_name}_trend_compare.txt")
    else:
        out_path = os.path.join(out_dir,f"{exp_name}_{stratified_term}_trend_compare.txt")
    
    if SAVE_INTERIM_FILE:
        f = open(out_path,"w").close()
        f = open(out_path,"a")
    else:
        f = open(os.devnull,'w')
    
    num_of_terms_changed = 0
    num_turning_points_changed = 0
    
    df1 = pd.read_csv(full_trend_csv)
    
    if empty_d2:
        f.close()
        num_of_terms_changed =  len(list(df1.term_raw.unique()))
        try: 
            df1_turning = df1[~df1['turning'].isna()]
            num_turning_points_changed  = df1_turning.shape[0]
        except KeyError:
            pass
        return f"{exp_name},1,1,,,"
        
    try:
        df2 = pd.read_csv(sample_trend_csv)
    except pd.errors.EmptyDataError:
        f.close()
        num_of_terms_changed =  len(list(df1.term_raw.unique()))
        try: 
            df1_turning = df1[~df1['turning'].isna()]
            num_turning_points_changed  = df1_turning.shape[0]
        except KeyError:
            pass
        return f"{exp_name},1,1,,,"
    
    
    
    term_to_check = df1.term_raw.unique()
    new_terms = df2.term_raw.unique()
    
    
    per_var_full_list = []
    per_var_sampled_list = []
    diff_per_var_list = []
    diff_overall_list = []
    per_diff_overall_list = []
    
    total_num_terms_in_full = len(list(term_to_check))
    df1_turning = df1[~df1['turning'].isna()]
    # total_num_turning_points_in_full = sum(1/df1_turning['weight'])
    full_r2 = full_trend_csv.replace("___trend.csv",".r2")
    sample_r2 = sample_trend_csv.replace("___trend.csv",".r2")
    
    
    with open(full_r2,"r") as full_r2_f:
        r2_full_val = full_r2_f.readlines()[0].strip()
    
    with open(sample_r2,"r") as sample_r2_f:
        sample_r2_lines = sample_r2_f.readlines()
        r2_sampled_val = sample_r2_lines[0].strip()
        if len(sample_r2_lines) > 1:
            r2_unsampled_val = sample_r2_lines[1].strip()
        else:
            r2_unsampled_val = ""
    
    for term in new_terms:
        if term not in term_to_check:
            num_of_terms_changed += 1
            df2_term = df2.loc[df2['term_raw']==term,:]
            term_weight = df2_term['weight'].iloc[0]
            try:
                df2_turning = df2_term['turning'].iloc[0]
                if not np.isnan(df2_turning):
                    num_turning_points_changed += 1/term_weight
            except KeyError:
                continue
            

    for term in term_to_check:
        print("------------------------------------------------------------------------------------",file=f)
        # term = term_to_check[1]
        if term not in df2.term_raw.unique():
            print(f"{term} no longer significant in sampled dataset",file=f)
            # print(f"{term} no longer significant in sampled dataset",file=f)
            num_of_terms_changed += 1
            df1_term = df1.loc[df1['term_raw']==term,:]
            term_weight = df1_term['weight'].iloc[0]
            try:
                df1_turning = df1_term['turning'].iloc[0]
                if not np.isnan(df1_turning):
                    num_turning_points_changed += 1/term_weight
            except KeyError:
                continue
        else:
            print(f"Checking term {term}:",file=f)
            df1_term = df1.loc[df1['term_raw']==term,:]
            df2_term = df2.loc[df2['term_raw']==term,:]
            print(f"Diff in {term}'s per_var_explained: {abs(df1_term.iloc[0,2]-df2_term.iloc[0,2]):.4f}",file=f)
            print(f"->Full:{df1_term.iloc[0,2]:.4f}, Sampled:{df2_term.iloc[0,2]:.4f}\n",file=f)
            diff_per_var_list.append(abs(df1_term.iloc[0,2]-df2_term.iloc[0,2]))
            per_var_full_list.append(df1_term.iloc[0,2])
            per_var_sampled_list.append(df2_term.iloc[0,2])
            
            if df1_term.shape[0] == 1:
                # print(f"Diff in {term}'s overall spearmanr: {abs(df1_term.iloc[0,5]-df2_term.iloc[0,5]):.4f}",file=f)
                # print(f"->Full:{df1_term.iloc[0,5]:.4f}, Sampled:{df2_term.iloc[0,5]:.4f}\n",file=f)
                diff_overall_list.append(abs(df1_term.iloc[0,5]-df2_term.iloc[0,5]))
                per_diff_overall_list.append(abs(df1_term.iloc[0,5]-df2_term.iloc[0,5])/abs(df1_term.iloc[0,5]))
                
                try:
                    df1_turning = df1_term['turning'].iloc[0]
                    df2_turning = df2_term['turning'].iloc[0]
                    term_weight = df1_term['weight'].iloc[0]
                except KeyError:
                    continue
                
                if np.isnan(df1_turning) and np.isnan(df2_turning):
                    continue
                elif not np.isnan(df1_turning) and not np.isnan(df2_turning):
                    print(f"Both have turning point:",file=f)
                    if not((df1_term['turning_spear2'].iloc[0] == -888 and df2_term['turning_spear2'].iloc[0] == -888) or (df1_term['turning_spear2'].iloc[0] != -888 and df2_term['turning_spear2'].iloc[0] != -888)):
                        num_turning_points_changed += 1/term_weight
                    # print(f"Turning point: {df1_turning}(full) vs {df2_turning}(sampled)")
                    # print(f"spear diff before turning:{abs(df1_term['turning_spear1'].iloc[0]-df2_term['turning_spear1'].iloc[0])}, after turning:{abs(df1_term['turning_spear2'].iloc[0]-df2_term['turning_spear2'].iloc[0])}")
                    print(f"Turning point in full dataset:{df1_turning}, spear1={df1_term['turning_spear1'].iloc[0]:.4f}, spear2={df1_term['turning_spear2'].iloc[0]:.4f}",file=f)
                    print(f"Turning point in sampled dataset:{df2_turning}, spear1={df2_term['turning_spear1'].iloc[0]:.4f}, spear2={df2_term['turning_spear2'].iloc[0]:.4f}",file=f)
                else:
                    num_turning_points_changed += 1/term_weight
                    print(f"One has turning point, one does not",file=f)
            else:
                diff_overall_group = []
                per_diff_overall_group = []
                
                df1_term_grp = df1_term.groupby(["fixed_parm_name","fixed_parm_val"])
                for (name,val),gp1 in df1_term_grp:
                    print(f"Fixing {name} @ {val}:",file=f)
                    gp2 = df2_term.loc[(df2_term['fixed_parm_name'] == name) & (df2_term['fixed_parm_val'] == val)]
                    if gp2.shape[0] == 0:
                        print("    Not found in sampled dataset!",file=f)
                        continue
                        
                    # print(f"    Diff in {term}'s overall spearmanr: {abs(gp1.iloc[0,5]-gp2.iloc[0,5]):.4f}",file=f)
                    # print(f"    ->Full dataset: {gp1.iloc[0,5]:.4f}, sampled dataset: {gp2.iloc[0,5]:.4f}",file=f)
                    
                    diff_overall_group.append(abs(gp1.iloc[0,5]-gp2.iloc[0,5]))
                    per_diff_overall_group.append(abs(gp1.iloc[0,5]-gp2.iloc[0,5]) / abs(gp1.iloc[0,5]))
                    
                    try:
                        df1_turning = gp1['turning'].iloc[0]
                        df2_turning = gp2['turning'].iloc[0]
                        term_weight = gp1['weight'].iloc[0]
                    except KeyError:
                        continue
                    
                    if np.isnan(df1_turning) and np.isnan(df2_turning):
                        continue
                    elif not np.isnan(df1_turning) and not np.isnan(df2_turning):
                        print(f"    Both have turning point:",file=f)
                        if not((gp1['turning_spear2'].iloc[0] == -888 and gp2['turning_spear2'].iloc[0] == -888) or (gp1['turning_spear2'].iloc[0] != -888 and gp2['turning_spear2'].iloc[0] != -888)):
                            num_turning_points_changed+= 1/term_weight
                        # print(f"    Turning point: {df1_turning}(full) vs {df2_turning}(sampled)")
                        # print(f"    spear diff before turning:{abs(gp1['turning_spear1'].iloc[0]-gp2['turning_spear1'].iloc[0])}, after turning:{abs(gp1['turning_spear2'].iloc[0]-gp2['turning_spear2'].iloc[0])}")
                        print(f"    Turning point in full dataset:{df1_turning}, spear1={gp1['turning_spear1'].iloc[0]:.4f}, spear2={gp1['turning_spear2'].iloc[0]:.4f}",file=f)
                        print(f"    Turning point in sampled dataset:{df2_turning}, spear1={gp2['turning_spear1'].iloc[0]:.4f}, spear2={gp2['turning_spear2'].iloc[0]:.4f}",file=f)
                    else:
                        num_turning_points_changed += 1/term_weight
                        print(f"    One has turning point, one does not",file=f)
                
                diff_overall_list.append(np.mean(diff_overall_group))
                per_diff_overall_list.append(np.mean(per_diff_overall_group))
        
        
    if SAVE_INTERIM_FILE:
        print("====================================================================================",file=f)
        print(f"***{num_of_terms_changed} terms are no longer significant in sampled dataset",file=f)
        print(f"***{num_turning_points_changed} turning points/knee points mismatch",file=f)
        print(f"***Mean difference in terms of per_var_explained:{np.mean(diff_per_var_list):.4f}",file=f)
        print(f"***R-squared in full dataset:{r2_full_val}",file=f)
        print(f"***R-squared of per_var_explained in sampled dataset:{r2_sampled_val}",file=f)
        
        print(f"***Mean difference in terms of overall spearmen rank correlation: {np.mean(diff_overall_list):.4f} ",file=f)
        print("====================================================================================",file=f)
    # print(f"***Mean %change in terms of overall spearmen rank correlation: {np.mean(per_diff_overall_list):.4f} ",file=f)
    
    
    f.close()
    
    if not SAVE_INTERIM_FILE:
        os.remove(sample_trend_csv)
        os.remove(sample_r2)
    
    return f"{exp_name},{num_of_terms_changed/total_num_terms_in_full},{num_turning_points_changed/total_num_terms_in_full},{np.mean(diff_per_var_list)},{r2_full_val},{r2_sampled_val},{r2_unsampled_val}"

def norm_0_1(x1,x2):
    
    n1 = (x1 - np.min(x1))/np.ptp(x1)
    n2 = (x2 - np.min(x2))/np.ptp(x2)
    
    return n1,n2

def calc_frechet(full_cof_csv,sample_cof_csv,name,out_dir,empty_d2 = False, stratified_term = False):  
# def calc_frechet(fname,DIR_NAME,DIR_NAME_SAMPLED,per_core=True):
    
    if not stratified_term:
        out_path = os.path.join(out_dir,f"{name}_trend_compare.txt")
    else:
        out_path = os.path.join(out_dir,f"{name}_{stratified_term}_trend_compare.txt")
    # f = open(out_path,"w").close()
    if SAVE_INTERIM_FILE:
        f = open(out_path,"a")
    else:
        f = open(os.devnull,"w")
    
    
    df1 = pd.read_csv(full_cof_csv)
    if empty_d2:
        f.close()
        return f""
    
    try:
        df2 = pd.read_csv(sample_cof_csv)
    except pd.errors.EmptyDataError:
        f.close()
        return f""
    
    term_to_check = df1.term_raw.unique()
    
    frechet_list = []
    
    for term in term_to_check:
        if term not in df2.term_raw.unique():
            continue
        else:
            print(f"Frechet distance for term {term}:",file=f)
            df1_term = df1.loc[df1['term_raw']==term,:]
            df2_term = df2.loc[df2['term_raw']==term,:]
            if df1_term['#parm'].iloc[0] == 1:
                coef1 = json.loads(df1_term['Coef'].iloc[0])
                coef2 = json.loads(df2_term['Coef'].iloc[0])
                
                parm1 = json.loads(df1_term['parm0_val'].iloc[0])
                parm2 = json.loads(df2_term['parm0_val'].iloc[0])
                
                coef1,coef2 = norm_0_1(coef1,coef2)
                parm1,parm2 = norm_0_1(parm1,parm2)
                
                c1 = np.column_stack((parm1,coef1))
                c2 = np.column_stack((parm2,coef2))
                fd = frechet_dist(c1,c2)
            
            
            else:
                assert df1_term['parm1'].iloc[0] == df2_term['parm1'].iloc[0]
                coef1 = json.loads(df1_term['Coef'].iloc[0])
                coef2 = json.loads(df2_term['Coef'].iloc[0])
                coef1,coef2 = norm_0_1(coef1,coef2)
                
                parm10 = json.loads(df1_term['parm1_val'].iloc[0])
                parm11 = json.loads(df1_term['parm2_val'].iloc[0])        
                
                parm20 = json.loads(df2_term['parm1_val'].iloc[0])
                parm21 = json.loads(df2_term['parm2_val'].iloc[0])  
                
                parm10,parm20 = norm_0_1(parm10,parm20)
                parm11,parm21 = norm_0_1(parm11,parm21)
                
                c1 = np.column_stack((parm10,parm11,coef1))
                c2 = np.column_stack((parm20,parm21,coef2))
                fd = frechet_dist(c1,c2,3) 
            
            print(f"Frechet distance = {fd}",file=f)
            frechet_list.append(fd)
    
    if not SAVE_INTERIM_FILE:
        os.remove(sample_cof_csv)
    
    return np.mean(frechet_list)

def run_all_baseline():
    input_base_dir = "/ANOVA/ExtensiveEval/csv"
    input_csvs_name = {
        "aria_tpcc.csv":"aria_tpcc_Aria",
        "calvin_tpcc.csv":"calvin_tpcc_Calvin",
        "cicada_ycsb.csv":"cicada_ycsb_Cicada",
        "drtm_tpcc.csv":"drtm_tpcc",
        "gam_tpcc.csv":"gam_tpcc",
        "herd_ycsb.csv":"herd_ycsb",
        "janus_tpcc.csv":"janus_tpcc_Janus",
        "mysql_ram_tpcc.csv":"mysql_tpcc",
        "silo_tpcc.csv":"silo_tpcc_Silo",
        "silo_ycsb.csv":"silo_ycsb",
        "star_tpcc.csv":"star_tpcc",
        "star_ycsb.csv":"star_ycsb",
        "tapir_ycsb.csv":"tapir_ycsb",
        "postgresql_ssd_tpcc.csv":"postgresql_tpcc"
    }
    print("Single terms first, followed by combined terms")
    print("System,max_terms_possible,num_significant_parms,num_turning_knee")

    works = []
    for csv_name,exp_name in input_csvs_name.items():
        df_raw = pd.read_csv(f"{input_base_dir}/{csv_name}")
        e = exp_name.split("_")
        if len(e) == 3:
            df, _ = preprocess(e[0], e[1], e[2], df_raw)
            exp_name_new = f"{e[0]}-{e[2]}_{e[1]}"
        else:
            df, _ = preprocess(e[0], e[1], None, df_raw)
            exp_name_new = f"{e[0]}_{e[1]}"
        
        works.append((df,exp_name_new))
    
    
    # with mp.Pool(len(works)) as p:
    #     p.map(stat_for_single_alg,works)
    for work in works:
        stat_for_single_alg(work)
        

def sample_all_systems(sample_method,random_seed=None):
    input_base_dir = "/ANOVA/ExtensiveEval/csv"
    input_csvs_name = {
        "aria_tpcc.csv":"aria_tpcc_Aria",
        "calvin_tpcc.csv":"calvin_tpcc_Calvin",
        "cicada_ycsb.csv":"cicada_ycsb_Cicada",
        "drtm_tpcc.csv":"drtm_tpcc",
        "gam_tpcc.csv":"gam_tpcc",
        "herd_ycsb.csv":"herd_ycsb",
        "janus_tpcc.csv":"janus_tpcc_Janus",
        "mysql_ram_tpcc.csv":"mysql_tpcc",
        "silo_tpcc.csv":"silo_tpcc_Silo",
        "silo_ycsb.csv":"silo_ycsb",
        "star_tpcc.csv":"star_tpcc",
        "star_ycsb.csv":"star_ycsb",
        "tapir_ycsb.csv":"tapir_ycsb",
        "postgresql_ssd_tpcc.csv":"postgresql_tpcc"
    }
    
    output_dir = f"/ANOVA/all_samples_{sample_method}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    processes = []
    
    for csv_name,exp_name in input_csvs_name.items():
        df_raw = pd.read_csv(f"{input_base_dir}/{csv_name}")
        e = exp_name.split("_")
        # sample(data, method, system, bench, output_dir=None, seed=None, alg=None)
        if len(e) == 3:
            processes.append(mp.Process(target = sample.sample, args = (df_raw, sample_method, e[0], e[1], output_dir, random_seed, e[2])))
        else:
            processes.append(mp.Process(target = sample.sample, args = (df_raw, sample_method, e[0], e[1], output_dir, random_seed, None)))

    for process in processes:
        process.start()
    
    for process in processes:
        process.join()
        
def run_one_comparison(sample_csv_path, sample_method, write_lock = None):
    csv_name = sample_csv_path.split('/')[-1]
    unsample_csv_path = sample_csv_path.replace("_exp.csv","_unexp.csv")
    
    info = {}
    for chunks in csv_name.split('_')[:-1]:
        if "@" in chunks:
            k,v = chunks.split("@")
            info[k] = v
        else:
            info['term'] += f"_{chunks}"
    
    
    exp_name_full = f"{info['sys']}_{info['bench']}"
    
    if sample_method ==  "stratified":
        exp_name = f"{exp_name_full}_{info['term']}"
    else:
        exp_name = exp_name_full
    
    result_dir_sampled = f"/ANOVA/{sample_method}_full_result/sample@{info['sample']}_random@{info['random']}_split@{info['split']}"
    
    with write_lock:
        if not os.path.exists(result_dir_sampled):
            os.makedirs(result_dir_sampled)
    
    try:
        stat_for_single_alg((pd.read_csv(sample_csv_path),exp_name,result_dir_sampled,pd.read_csv(unsample_csv_path)))
    except ValueError as ex:
        return  

    assert os.path.exists(os.path.join(result_dir_sampled,f"{exp_name}_stopped_at"))
    
    empty_d2 = False
    if not os.path.exists(os.path.join(result_dir_sampled,f"{exp_name}___trend.csv")):
        empty_d2 = True
    with open(os.path.join(result_dir_sampled,f"{exp_name}_stopped_at"),"r") as ff:
        stopped_at = ff.readlines()[0].strip()
    
    if stopped_at == "2":
        result_dir_full = "/ANOVA/ANOVA_result_full_COMBINE"
    else:
        result_dir_full = "/ANOVA/ANOVA_result_full_SINGLE"
        
    trend1 = compare_trend(os.path.join(result_dir_full,f"{exp_name_full}___trend.csv"),os.path.join(result_dir_sampled,f"{exp_name}___trend.csv"),exp_name,result_dir_sampled,empty_d2=empty_d2)
    trend2 = calc_frechet(os.path.join(result_dir_full,f"{exp_name_full}___cof.csv"),os.path.join(result_dir_sampled,f"{exp_name}___cof.csv"),exp_name,result_dir_sampled,empty_d2=empty_d2)
    
    with write_lock:
        with open(f"/ANOVA/RESULT_{sample_method}.csv","a") as f:
            if sample_method == "stratified":
                print(f"{info['term']},{info['sample']},{info['random']},{info['split']},{trend1},{trend2}",file=f)
            else:
                print(f"{info['sample']},{info['random']},{info['split']},{trend1},{trend2}",file=f)

    if not SAVE_INTERIM_FILE:
        os.remove(sample_csv_path)
        os.remove(unsample_csv_path)
        os.remove(os.path.join(result_dir_sampled,f"{exp_name}_stopped_at"))


def run_all_comparsion(sample_method):
    print(f"Running sampling with: {sample_method}")
    print("Current time: %s" %time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    sample_all_systems(sample_method)
    print(f"Done!")
    print("Current time: %s" %time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    print("Start comparing.........")
    with open(f"/ANOVA/RESULT_{sample_method}.csv","w") as f:
        if sample_method == "stratified":
            print("stratified_term,sample_ratio,random,split,alg,num_of_terms_changed,num_turning_mismatch,avg_diff_per_var,r2_full,r2_sample,r2_unsample,avg_frechet_dist",file=f)
        else:
            print("sample_ratio,random,split,alg,num_of_terms_changed,num_turning_mismatch,avg_diff_per_var,r2_full,r2_sample,r2_unsample,avg_frechet_dist",file=f)
    
    lock = mp.Lock()
    csvs_path = glob.glob(f"/ANOVA/all_samples_{sample_method}/*_exp.csv")
    
    chunker = lambda seq, size: (seq[pos:pos + size] for pos in range(0, len(seq), size))
    
    cpus_to_use = os.cpu_count()
    total_jobs = len(csvs_path) // cpus_to_use + 1
    done_jobs = 0
    one_percent_jobs = total_jobs // 100
    
    for work_list in chunker(csvs_path,os.cpu_count()):
        
        if done_jobs % one_percent_jobs == 0:
            percent_done = done_jobs // one_percent_jobs
            print(f"{percent_done}% done")
            print("Current time: %s" %time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        
        processes = [mp.Process(target=run_one_comparison,args=(csv,sample_method, lock)) for csv in work_list]
        
        for process in processes:
            process.start()
        
        for process in processes:
            process.join()
        
        done_jobs += 1
    
    print("Done!")
    print("Current time: %s" %time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    print("----------------------------------------------------------------------------")
    # work_list = csvs_path[:100]
    # for work in work_list:
    #     run_one_comparison(work,sample_method, lock)


def run_calvin_comparison():
    original_data = pd.read_csv("/ANOVA/ExtensiveEval/csv/calvin_tpcc.csv")
    revised_data = pd.read_csv("/ANOVA/ExtensiveEval/csv/aria_tpcc.csv")

    original_data = original_data[original_data["nnodes"] == 1]
    original_data = original_data[original_data["alg"] == "Calvin"]
    original_data.drop(columns=["nnodes", "alg"], inplace=True)

    revised_data = revised_data[revised_data["alg"] == "Calvin-1"]
    revised_data = revised_data[revised_data["threads"] == 4]
    revised_data.drop(columns=["alg", "threads"], inplace=True)
    revised_data.rename(columns={"Distributed": "dist"}, inplace=True)
    conditions = (revised_data["WH"] == 256) | (revised_data["WH"] == 512) | (revised_data["WH"] == 1024)
    revised_data = revised_data[~conditions]
    
    output_dir = f"/ANOVA/all_samples_calvin-comp"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # run random sampling
    random_sampling(original_data,  f"{output_dir}/method@random_sys@calvin-Calvin-comp_bench@tpcc", seed=None)
    random_sampling(revised_data,  f"{output_dir}/method@random_sys@aria-Calvin-comp_bench@tpcc", seed=None)
    
    #run baseline
    stat_for_single_alg((original_data,"calvin-Calvin-comp_tpcc"))
    stat_for_single_alg((original_data,"aria-Calvin-comp_tpcc"))
    
    
                        
    works = []
    
    
    sample_method = "calvin-comp"
    with open(f"/ANOVA/RESULT_{sample_method}.csv","w") as f:
        print("sample_ratio,random,split,alg,num_of_terms_changed,num_turning_mismatch,avg_diff_per_var,r2_full,r2_sample,r2_unsample,avg_frechet_dist",file=f)
    
    lock = mp.Lock()
    csvs_path = glob.glob(f"/ANOVA/all_samples_{sample_method}/*_exp.csv")
    
    chunker = lambda seq, size: (seq[pos:pos + size] for pos in range(0, len(seq), size))
    
    for work_list in chunker(csvs_path,32):
        processes = [mp.Process(target=run_one_comparison,args=(csv,sample_method, lock)) for csv in work_list]
        
        for process in processes:
            process.start()
        
        for process in processes:
            process.join()
    
    
    

if __name__ == "__main__":
    
    mp.set_start_method('forkserver')
    
    run_all_baseline()
    run_all_comparsion("random")
    run_all_comparsion("balance")
    run_all_comparsion("dist-aware")
    run_all_comparsion("stratified")
    # run_calvin_comparison()
    
    
    
    
    
