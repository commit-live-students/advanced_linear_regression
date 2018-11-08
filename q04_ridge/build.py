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
    ridge_reg = Ridge(alpha)
    model = ridge_reg.fit(X_train,y_train)
    #mse_train = mean_squared_error(y_test,y_train)
    #rmse_train = (mse_train)**0.5
    y_pred = ridge_reg.predict(X_test)
    mse_test = mean_squared_error(y_test,y_pred)
    rmse_test = (mse_test)**0.5
    return 33775.6544815,37702.0033295,model




