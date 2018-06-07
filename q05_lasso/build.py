# %load q05_lasso/build.py
# Default imports
from sklearn.linear_model import Lasso
import pandas as pd
import numpy as np
import math
from sklearn.metrics import mean_squared_error
from greyatomlib.advanced_linear_regression.q01_load_data.build import load_data

# We have already loaded the data for you
data_set, X_train, X_test, y_train, y_test = load_data('data/house_prices_multivariate.csv')

np.random.seed(9)


# Write your solution here
def lasso(alpha=0.01):
    lasso_reg_model = Lasso(alpha=alpha, normalize =True)
    lasso_reg_model.fit(X_train, y_train)
    
    y_train_pred = lasso_reg_model.predict(X_train)
    y_test_pred = lasso_reg_model.predict(X_test)
    rmse_train = math.sqrt(mean_squared_error(y_train_pred, y_train))
    rmse_test = math.sqrt(mean_squared_error(y_test_pred, y_test))
    return rmse_train, rmse_test

#lasso()

