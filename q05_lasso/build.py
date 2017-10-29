# %load q05_lasso/build.py
# Default imports
from sklearn.linear_model import Lasso
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from greyatomlib.advanced_linear_regression.q01_load_data.build import load_data

# We have already loaded the data for you
data_set, X_train, X_test, y_train, y_test = load_data('data/house_prices_multivariate.csv')
def lasso(alpha=0.01):
    x=Lasso(alpha=alpha, fit_intercept=True, normalize=True, copy_X=True, random_state=9)
    x.fit(X_train,y_train)
    y_pred = x.predict(X_train)
    rmse_train=mean_squared_error(y_train,y_pred)
    y_pred1=x.predict(X_test)
    rmse_test=mean_squared_error(y_test,y_pred1)
    return  np.sqrt(rmse_train),np.sqrt(rmse_test)

# Write your solution here
