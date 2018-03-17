# Default imports
from sklearn.linear_model import Lasso
import pandas as pd
import numpy as np
import math
from sklearn.metrics import mean_squared_error
from greyatomlib.advanced_linear_regression.q01_load_data.build import load_data
np.random.seed(9)

# We have already loaded the data for you
data_set, X_train, X_test, y_train, y_test = load_data('data/house_prices_multivariate.csv')


# Write your solution here
# alpha is learning rate
def lasso(alpha=0.01):
    lasso_regression_model = Lasso(alpha=alpha,normalize=True, random_state=9)
    lasso_regression_model.fit(X_train, y_train)
    # root mean squared error for train
    y_train_pred = lasso_regression_model.predict(X_train)
    y_test_pred = lasso_regression_model.predict(X_test)
    rmse_train = math.sqrt(mean_squared_error(y_train_pred, y_train))
    rmse_test = math.sqrt(mean_squared_error(y_test_pred, y_test))
    return (rmse_train, rmse_test)
