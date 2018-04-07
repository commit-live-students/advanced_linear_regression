# %load q04_ridge/build.py
# Default imports
from sklearn.linear_model import Ridge
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from greyatomlib.advanced_linear_regression.q01_load_data.build import load_data
from scipy import sqrt as sqrt

# We have already loaded the data for you
data_set, X_train, X_test, y_train, y_test = load_data('data/house_prices_multivariate.csv')

np.random.seed(9)


def ridge(alpha = 0.01):
    clf = Ridge(alpha=alpha,normalize=True,random_state=9)
    model = clf.fit(X_train,y_train)
    y_pred = clf.predict(X_train)
    y_pred1 = clf.predict(X_test)
    rmse_Train = sqrt(mean_squared_error(y_pred = y_pred, y_true = y_train))
    rmse_Test = sqrt(mean_squared_error(y_pred = y_pred1, y_true = y_test))
    return rmse_Train, rmse_Test

ridge(alpha = 0.01)

