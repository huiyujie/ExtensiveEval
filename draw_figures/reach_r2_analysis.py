import sys
sys.path.append("..")
import util_func
from draw_figures.read_results import get_anova_results, get_results

if __name__ == "__main__":
    combs = sorted(util_func.ALL_SYS)
    sample_method = "random"
    threshold = 0.9
    for comb in combs:
        filename = comb[0] + "-" + comb[1] + "-" + comb[2] if len(comb) == 3 else comb[0] + "-" + comb[1]
        anova_res = get_anova_results(filename, sample_method)
        ml_res = get_results(f"../results/ML/{sample_method}/{filename}.csv")
        anova_idx = (anova_res["p50"] >= threshold).idxmax()
        ml_idx = (ml_res["p50"] >= threshold).idxmax()
        output_name = f"{comb[0]}-{comb[1]}".capitalize()
        if anova_idx == 0 and ml_idx == 0:
            print(f"{output_name} (None)")
        elif anova_idx == 0:
            print(f"{output_name} (ML)")
        elif ml_idx == 0:
            print(f"{output_name} (ANOVA)")
        elif anova_idx < ml_idx:
            print(f"{output_name} (ANOVA)")
        elif anova_idx > ml_idx:
            print(f"{output_name} (ML)")
        else:
            r2_anova = anova_res["p50"][(ml_res["p50"] >= 0.9).idxmax()]
            ml_anova = ml_res["p50"][(ml_res["p50"] >= 0.9).idxmax()]
            if r2_anova > ml_anova:
                print(f"{output_name} (ANOVA Both)")
            else:
                print(f"{output_name} (ML Both)")