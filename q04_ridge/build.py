# %load q04_ridge/build.py
# Default imports
from sklearn.linear_model import Ridge
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from greyatomlib.advanced_linear_regression.q01_load_data.build import load_data

# We have already loaded the data for you
data_set, X_train, X_test, y_train, y_test = load_data('data/house_prices_multivariate.csv')

# Write your solution here
def ridge(alpha=0.01):
    np.random.seed(9)
    model = Ridge(alpha=alpha,fit_intercept=True,normalize=True,random_state=9)
    
    model.fit(X_train,y_train)
    yhat = model.predict(X_test)
    yhat1 = model.predict(X_train)    
    RMSE_for_train = np.sqrt(mean_squared_error(y_train,yhat1))
    RMSE_for_test = np.sqrt(mean_squared_error(y_test,yhat))      
    
    return RMSE_for_train,RMSE_for_test, model
ridge()

