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
def lasso(alpha=0.01):
    data_set, X_train, X_test, y_train, y_test = load_data('data/house_prices_multivariate.csv', random_state=9)
    lasso = Lasso(alpha=alpha, normalize=True)
    lasso.fit(X_train, y_train)
    
    y_pred_train = lasso.predict(X_train)
    y_pred_test = lasso.predict(X_test)
    
    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
    
    return rmse_train, rmse_test



