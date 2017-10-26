# Default imports
from sklearn.linear_model import Lasso
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from greyatomlib.advanced_linear_regression.q01_load_data.build import load_data
np.random.seed(9)

# We have already loaded the data for you
data_set, X_train, X_test, y_train, y_test = load_data('data/house_prices_multivariate.csv')

def lasso(alpha=0.01):
    model = Lasso(alpha=alpha,random_state=9,normalize=True)
    model.fit(X_train,y_train)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    rmse_train = mean_squared_error(y_train_pred,y_train)**0.5
    rmse_test = mean_squared_error(y_test_pred,y_test)**0.5
    return rmse_train,rmse_test

# Write your solution here
