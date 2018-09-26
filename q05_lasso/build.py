# %load q05_lasso/build.py
# Default imports
from sklearn import linear_model
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from greyatomlib.advanced_linear_regression.q01_load_data.build import load_data
from math import sqrt

# We have already loaded the data for you
a = 0.01
def lasso(alpha = 0.01 ):
    data_set, X_train, X_test, y_train, y_test = load_data('data/house_prices_multivariate.csv')
    #np.random.seed(9)
    lasso_model=linear_model.Lasso(alpha = alpha)
    Model = lasso_model.fit(X_train, y_train)
    y_pred_1 = lasso_model.predict(X_train)
    y_pred_2 = lasso_model.predict(X_test)
    rmse1 = float(sqrt(mean_squared_error(y_pred_1, y_train)))
    rmse2 = float(sqrt(mean_squared_error(y_pred_2, y_test)))
    return rmse1, rmse2, Model
lasso(0.01)






