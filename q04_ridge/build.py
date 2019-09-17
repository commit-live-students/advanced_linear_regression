# Default imports
from sklearn.linear_model import Ridge
import pandas as pd
import numpy as np
from math import sqrt
from sklearn.metrics import mean_squared_error
from greyatomlib.advanced_linear_regression.q01_load_data.build import load_data
np.random.seed(9)

# We have already loaded the data for you
data_set, X_train, X_test, y_train, y_test = load_data('data/house_prices_multivariate.csv')


# Write your solution here
def ridge(alpha=0.01):
    ridge_model=Ridge(alpha=alpha, max_iter=100000, normalize=True, random_state=9)
    ridge_model.fit(X_train, y_train)

    y_pred_test = ridge_model.predict(X_test)
    RMSE_test = sqrt(mean_squared_error(y_test, y_pred_test))

    y_pred_train = ridge_model.predict(X_train)
    RMSE_train = sqrt(mean_squared_error(y_train, y_pred_train))

    return RMSE_train, RMSE_test
