# %load q05_lasso/build.py
# Default imports
from sklearn.linear_model import Lasso
import pandas as pd
import numpy as np
import math
from sklearn.metrics import mean_squared_error
from greyatomlib.advanced_linear_regression.q01_load_data.build import load_data

# We have already loaded the data for you
data_set, X_train, X_test, y_train, y_test = load_data('data/house_prices_multivariate.csv')

np.random.seed(9)


# Write your solution here
def lasso(alpha=0.01):
    Model = Lasso(alpha,normalize=True, random_state=9)
    Model.fit(X_train, y_train)

    y_train_pred = Model.predict(X_train)
    train_mse = mean_squared_error(y_train, y_train_pred)
    train_rmse = math.sqrt(train_mse)
    
    y_pred_test = Model.predict(X_test)
    test_mse = mean_squared_error(y_test, y_pred_test)
    test_rmse = math.sqrt(test_mse)
    
    return train_rmse,test_rmse
    


