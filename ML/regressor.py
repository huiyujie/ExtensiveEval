from sklearn.metrics import r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV


def standarize(train, valid):
    global scaler
    scaler = StandardScaler()
    scaler.fit(train)
    return scaler.transform(train), scaler.transform(valid)

def process_data(train, valid, para_names):
    x_train = train[para_names]
    x_valid = valid[para_names]
    y_train = train["tput"].to_numpy()
    y_valid = valid["tput"].to_numpy()
    x_train, x_valid = standarize(x_train, x_valid)

    return x_train, y_train, x_valid, y_valid

def MLP_regression(train, valid, para_names):
    x_train, y_train, x_valid, y_valid = process_data(train, valid, para_names)
    regr = MLPRegressor(solver="lbfgs", hidden_layer_sizes=(10,10,5),learning_rate_init=1e-1, random_state=1, max_iter=10000).fit(x_train, y_train)
    y_pred = regr.predict(x_valid)
    r2 = r2_score(y_valid, y_pred)
    return r2

def Lasso_regression(train, valid, para_names):
    x_train, y_train, x_valid, y_valid = process_data(train, valid, para_names)
    # Use LassoCV to search for the best alpha parameter using 5-fold cross-validation
    lasso_cv = LassoCV(cv=5, random_state=1, max_iter=1000).fit(x_train, y_train)
    best_alpha = lasso_cv.alpha_
    # Fit the final Lasso model with the best found alpha
    regr = Lasso(alpha=best_alpha, random_state=1, max_iter=1000).fit(x_train, y_train)
    y_pred = regr.predict(x_valid)
    r2 = r2_score(y_valid, y_pred)
    return r2