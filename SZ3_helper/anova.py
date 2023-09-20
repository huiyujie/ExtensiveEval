from itertools import combinations

from sklearn.metrics import r2_score
from statsmodels.formula.api import ols
import statsmodels.api as sm
from parse_results import get_global_index


def anova(original, dim_info, samples):
    predicting_data = set_samples(original, samples, dim_info.get_dimensions())
    factor_names = dim_info.get_factor_names()
    train, valid = extract_sample_train(predicting_data)
    train.drop(columns=["is_test", "is_filled", "_merge"], inplace=True)
    valid.drop(columns=["is_test", "is_filled", "_merge"], inplace=True)

    for f in factor_names:
        print(f)
        print(train[f].unique().tolist())

    try:
        r2 = anova_regression(train, valid, factor_names, 1)
    except Exception as e:
        print(e)
        r2 = 0
    return r2


def set_samples(predict_data, samples, dims):
    predict_data["is_test"] = True
    for sample in samples:
        offset = get_global_index(sample, dims)
        predict_data.loc[offset, "is_test"] = False

    return predict_data

def extract_sample_train(predict_data):
    predict_data = predict_data[predict_data["is_filled"] == False]
    train = predict_data[predict_data["is_test"] == False]
    valid = predict_data[predict_data["is_test"] == True]

    return train, valid


def anova_regression(train, valid, param_names, degree):
    eq, _ = remove_terms(param_names, train, degree, False)
    model = ols(eq, data=train).fit()

    y_pred = model.predict(valid)

    try:
        r2 = r2_score(valid["tput"], y_pred)
    except Exception as e:
        print(valid)
        print("valid data")
    return r2

def remove_terms(params, dataframe, degree, log):
    terms = init_terms(params, degree)
    remove = ["-1"]
    threshold = 0.05
    removed_terms = []

    # nan_cnt = sum(np.isnan(anova_table_init["PR(>F)"].to_numpy()))
    pvalue_dict = dict()
    while len(remove) != 0:
        eq = format_equation(terms)
        model = ols(eq, data=dataframe).fit()
        anova_table = sm.stats.anova_lm(model, typ=3)
        anova_table = anova_table[~anova_table.index.isin(["Intercept"])]
        invalid = list(anova_table[anova_table["PR(>F)"] > threshold].index)

        if len(invalid) > 0:
            remove = anova_table["PR(>F)"].idxmax()
            removed_row = anova_table["PR(>F)"].loc[remove]
            pvalue_dict[remove] = removed_row
            removed_terms.append(remove)
            terms.remove(remove)
        else:
            break

    if log:
        print("Removed all the insignificant terms.")
        # print(model.summary())
        print(anova_table)
        # print("most significant is " + str(anova_table["PR(>F)"].idxmin()))
        print("Removed terms: " + str(removed_terms))
        # print("most significant " + anova_table_init["PR(>F)"].idxmin())
    return eq, pvalue_dict


def init_terms(terms, degree=1):
    initial_term = []
    initial_term.append(list(combinations(terms, 1)))
    if degree == 2:
        initial_term.append(list(combinations(terms, 2)))

    new_terms = []
    for term in initial_term[0]:
        new_terms.append("C(" + term[0] + ")")
    if degree == 2:
        for term in initial_term[1]:
            new_terms.append("C(" + term[0] + "):C(" + term[1] + ")")
    return new_terms

def format_equation(terms):
    equation = "tput~"

    for term in terms:
        equation += term + "+"
    equation = equation[:-1]
    return equation