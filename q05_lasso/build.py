from sklearn.linear_model import Lasso
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from greyatomlib.advanced_linear_regression.q01_load_data.build import load_data
from math import sqrt
np.random.seed(9)
np.random.RandomState(9)

# We have already loaded the data for you
data_set, X_train, X_test, y_train, y_test = load_data('data/house_prices_multivariate.csv')

# Write your solution here
def lasso(alpha=0.01):
    lasso_model = Lasso(alpha=alpha, normalize=True)
    lasso_model.fit(X_train, y_train)
    train_yprediction = lasso_model.predict(X_train)
    test_yprediction = lasso_model.predict(X_test)
    rmse_train = sqrt(mean_squared_error(train_yprediction, y_train) )
    rmse_test = sqrt(mean_squared_error(test_yprediction, y_test))
    return rmse_train, rmse_test
