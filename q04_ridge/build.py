# %load q04_ridge/build.py
# Default imports
from sklearn.linear_model import Ridge
import pandas as pd
import numpy as np
from math import sqrt
from sklearn.metrics import mean_squared_error
from greyatomlib.advanced_linear_regression.q01_load_data.build import load_data

# We have already loaded the data for you
data_set, X_train, X_test, y_train, y_test = load_data('data/house_prices_multivariate.csv')

np.random.seed(9)


# Write your solution here
def ridge(alpha=0.01):
    
    model = Ridge(alpha=alpha, random_state=9,normalize=True)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    y_pred_train = model.predict(X_train)
    return sqrt(mean_squared_error(y_train, y_pred_train)), sqrt(mean_squared_error(y_test, y_pred)), model.fit(X_train, y_train), 
ridge()

 



