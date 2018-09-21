# %load q04_ridge/build.py
# Default imports
from sklearn.linear_model import Ridge
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from greyatomlib.advanced_linear_regression.q01_load_data.build import load_data

# We have already loaded the data for you
data_set, X_train, X_test, y_train, y_test = load_data('data/house_prices_multivariate.csv')

np.random.seed(9)

# Write your solution here
def ridge(alpha = 0.01):
    ridge_model = Ridge(normalize = True,alpha = 0.01,random_state = 9)
    ridge_model.fit(X_train,y_train)
    y_train_ridge = ridge_model.predict(X_train)
    y_test_ridge = ridge_model.predict(X_test)
    Rmse_train = mean_squared_error(y_train,y_train_ridge)**0.5
    Rmse_test = mean_squared_error(y_test,y_test_ridge)**0.5
    return Rmse_train,Rmse_test,ridge_model
ridge()


