# %load q04_ridge/build.py
# Default imports
from sklearn.linear_model import Ridge
import pandas as pd
import numpy as np
import math
from sklearn.metrics import mean_squared_error
from greyatomlib.advanced_linear_regression.q01_load_data.build import load_data

# We have already loaded the data for you
data_set, X_train, X_test, y_train, y_test = load_data('data/house_prices_multivariate.csv')

np.random.seed(9)


# Write your solution here
def ridge(alpha=0.01):
    ridge_reg_model = Ridge(alpha=alpha,normalize=True)
    ridge_reg_model.fit(X_train, y_train)
    
    y_train_pred = ridge_reg_model.predict(X_train)
    y_test_pred = ridge_reg_model.predict(X_test)
    rmse_train = math.sqrt(mean_squared_error(y_train_pred,y_train))
    rmse_test = math.sqrt(mean_squared_error(y_test_pred,y_test))
    return rmse_train, rmse_test, ridge_reg_model

#ridge()


