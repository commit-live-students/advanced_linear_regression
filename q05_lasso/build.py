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
def lasso(alpha=.01):
    #Model
    l1=Lasso(alpha=alpha,normalize=True,random_state=9)
    l1.fit(X_train,y_train)
    
    #RMSE & Prediction
    prediction=l1.predict(X_train)
    rmse1=mean_squared_error(prediction,y_train)**.5
    
    prediction=l1.predict(X_test)
    rmse2=mean_squared_error(prediction,y_test)**.5
    
    return rmse1,rmse2


