# %load q04_ridge/build.py
# Default imports
from sklearn.linear_model import Ridge
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from greyatomlib.advanced_linear_regression.q01_load_data.build import load_data
import math
# We have already loaded the data for you
data_set, X_train, X_test, y_train, y_test = load_data('data/house_prices_multivariate.csv')

np.random.seed(9)

def ridge(alpha=0.01):
    ridgeReg = Ridge(alpha=alpha , normalize=True,random_state=9)
    ridgeReg.fit(X_train,y_train)
    ypred_train= ridgeReg.predict(X_train)
    
    #calculate RMSE for training data
    rmse_train = math.sqrt(mean_squared_error(y_train, ypred_train))
    
    #calculate Rmse for testing data
    ypred_test= ridgeReg.predict(X_test)
    rmse_test = math.sqrt(mean_squared_error(y_test, ypred_test))
    
    return rmse_train,rmse_test, ridgeReg


ridge(alpha=0.01)


