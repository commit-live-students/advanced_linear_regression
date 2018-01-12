# %load q05_lasso/build.py
# Default imports
from sklearn.linear_model import Lasso
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from greyatomlib.advanced_linear_regression.q01_load_data.build import load_data

# We have already loaded the data for you
data_set, X_train, X_test, y_train, y_test = load_data('data/house_prices_multivariate.csv')

np.random.seed(9)


# Write your solution here
from sklearn.linear_model import Ridge
def lasso(alpha=0.01):
    ridge_model = Lasso(alpha=alpha, normalize=True, random_state=9)
    m = ridge_model.fit(X_train, y_train)
    y_pred_train = m.predict(X_train)
    y_pred_test = m.predict(X_test)
    RMSE_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
    RMSE_test = np.sqrt(mean_squared_error(y_test,y_pred_test))
    return RMSE_train, RMSE_test

rmsetrain, rmsetest = lasso()
print "RMSE Train: {0}:  dtype: {1}".format(rmsetrain,type(rmsetrain))
print "RMSE Test: {0}:  dtype: {1}".format(rmsetest,type(rmsetest))
