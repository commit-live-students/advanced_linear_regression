# %load q05_lasso/build.py
# Default imports
from sklearn.linear_model import Lasso
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from greyatomlib.advanced_linear_regression.q01_load_data.build import load_data
from scipy import sqrt as sqrt
# We have already loaded the data for you
data_set, X_train, X_test, y_train, y_test = load_data('data/house_prices_multivariate.csv')

np.random.seed(9)

def lasso(alpha = 0.01):
    clf = Lasso(alpha = alpha, normalize=True, random_state=9)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_train)
    y_pred1 = clf.predict(X_test)
    rmse = sqrt(mean_squared_error(y_true = y_train, y_pred = y_pred))
    rmse1 = sqrt(mean_squared_error(y_true = y_test, y_pred = y_pred1))
    return rmse, rmse1
lasso(alpha = 0.01)

