# %load q05_lasso/build.py
# Default imports
from sklearn.linear_model import Lasso
import pandas as pd
import numpy as np
from math import sqrt
from sklearn.metrics import mean_squared_error
from greyatomlib.advanced_linear_regression.q01_load_data.build import load_data

# We have already loaded the data for you
data_set, X_train, X_test, y_train, y_test = load_data('data/house_prices_multivariate.csv')

np.random.seed(9)

def lasso(alpha=0.01):
    lasso_model=Lasso(alpha=0.01, max_iter=1000, random_state=9,normalize=True)
    lasso_model.fit(X_train, y_train)
    y_pred = lasso_model.predict(X_test)
    y_pred_train = lasso_model.predict(X_train)
    rmsetest=sqrt(mean_squared_error(y_test, y_pred))
    rmsetrain=sqrt(mean_squared_error(y_train, y_pred_train))
    return rmsetrain,rmsetest

# rmsetrain,rmsetest=lasso()
# print(rmsetrain)
# print(rmsetest)
# print(type(rmsetrain))
# print(type(rmsetest))


# Write your solution here
