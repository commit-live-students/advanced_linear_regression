# %load q04_ridge/build.py
# Default imports
from sklearn.linear_model import Ridge
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from greyatomlib.advanced_linear_regression.q01_load_data.build import load_data

# We have already loaded the data for you
data_set, X_train, X_test, y_train, y_test = load_data('data/house_prices_multivariate.csv')

# np.random.seed(9)


#Write your solution here
def ridge(alpha = 0.01):
    ridge_mod = Ridge(alpha,normalize=True, random_state=9)
    ridge_mod.fit(X_train, y_train)
    y_pred = ridge_mod.predict(X_test)
    y_tran = ridge_mod.predict(X_train)
    RMSE_train = mean_squared_error(y_tran,y_train)**0.5
    RMSE_test  = mean_squared_error(y_test,y_pred)**0.5
    
    return RMSE_train,RMSE_test,ridge_mod



ridge(alpha = 0.01)





