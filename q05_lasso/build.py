# Default imports
from sklearn.linear_model import Lasso
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from greyatomlib.advanced_linear_regression.q01_load_data.build import load_data
np.random.seed(9)

# We have already loaded the data for you
data_set, X_train, X_test, y_train, y_test = load_data('data/house_prices_multivariate.csv')


# Write your solution here
def lasso(learning_rate=0.01):
    clf = Lasso(alpha=learning_rate, normalize=True, random_state=9)
    clf.fit(X_train, y_train)
    y_pred_tr = clf.predict(X_train)
    y_pred_te = clf.predict(X_test)
    rmse_tr = mean_squared_error(y_train, y_pred_tr)
    rmse_te = mean_squared_error(y_test,  y_pred_te)
    return np.sqrt(rmse_tr), np.sqrt(rmse_te) 
