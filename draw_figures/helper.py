from draw_figures.read_results import get_results


# pick the best results from stratified sampling
def pick_best_stratified(filename, factor_names):
    datas = {}
    max = 0
    for factor in factor_names:
        data = get_results(f"../results/XGBoost/stratified/{filename}-{factor}.csv")
        datas[factor] = data
        if data["p50"].median() > max:
            max = data["p50"].median()
            best = factor
    
    return best

# pick the best results from stratified sampling
def pick_best_neyman(filename, factor_names):
    datas = {}
    max = 0
    for factor in factor_names:
        data = get_results(f"../results/XGBoost/stratified/{filename}-{factor}.csv")
        datas[factor] = data
        if data["p50"].median() > max:
            max = data["p50"].median()
            best = factor
    
    return best


