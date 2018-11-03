# %load q04_ridge/build.py
# Default imports
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from greyatomlib.advanced_linear_regression.q01_load_data.build import load_data
from sklearn.pipeline import make_pipeline
# We have already loaded the data for you
data_set, X_train, X_test, y_train, y_test = load_data('data/house_prices_multivariate.csv')

np.random.seed(9)

def ridge(alpha = 0.01):
    model = Ridge(alpha = alpha , normalize  = True)
    model.fit(X_train , y_train)
    #model.fit(X_train, y_train)
    y = model.predict(X_train)
    yt = model.predict(X_test)
    RMSE =  mean_squared_error (y,y_train)
    RMSE1 =  mean_squared_error (y_test,yt)
    return np.sqrt(RMSE), np.sqrt(RMSE1) ,model 



