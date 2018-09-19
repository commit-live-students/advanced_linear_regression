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

def lasso(alpha = 0.01):
    model = Lasso(alpha = alpha, random_state=9, normalize=True)
    model.fit(X_train, y_train)
    
    y_predict_test  = model.predict(X_test)
    y_predict_train = model.predict(X_train)
    
    rmse_train = mean_squared_error(y_train,y_predict_train)**0.5
    rmse_test  = mean_squared_error(y_test,y_predict_test)**0.5

    #print (rmse_train)
    #print (rmse_test)
    #print (model)
    
    return rmse_train,rmse_test

#lasso(alpha = 0.01)


