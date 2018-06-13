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
from math import sqrt

def lasso(alpha=0.01):
    model = Lasso(alpha, normalize = True, random_state= 9)
    lasso = model.fit(X_train,y_train)
    rmse_train = sqrt( mean_squared_error(y_train,  model.predict(X_train)))
    rmse_test = sqrt( mean_squared_error(y_test,  model.predict(X_test)))
    return (rmse_train,rmse_test)


