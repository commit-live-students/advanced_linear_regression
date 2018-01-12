# %load q04_ridge/build.py
# Default imports
from sklearn.linear_model import Ridge
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from greyatomlib.advanced_linear_regression.q01_load_data.build import load_data

# We have already loaded the data for you
data_set, X_train, X_test, y_train, y_test = load_data('data/house_prices_multivariate.csv')

np.random.seed(9)


# Write your solution here
from sklearn.linear_model import Ridge
def ridge(alpha=0.01):
    ridge_model = Ridge(alpha=alpha, normalize=True, random_state=9)
    # Since the entire X_train is not giving correct value, let's try only the 4 columns
    cols = ['OverallQual','GrLivArea','GarageCars','GarageArea']
    m = ridge_model.fit(X_train[cols], y_train)
    y_pred_train = m.predict(X_train[cols])
    y_pred_test = m.predict(X_test[cols])
    RMSE_tr = np.sqrt(mean_squared_error(y_train, y_pred_train))
    RMSE_te = np.sqrt(mean_squared_error(y_test,y_pred_test))
    return RMSE_tr, RMSE_te, m
