# Default imports
from sklearn.linear_model import Ridge
import pandas as pd
import numpy as np
import math
from sklearn.metrics import mean_squared_error
from greyatomlib.advanced_linear_regression.q01_load_data.build import load_data
np.random.seed(9)

# We have already loaded the data for you
data_set, X_train, X_test, y_train, y_test = load_data('data/house_prices_multivariate.csv')


def ridge(alpha = 0.01):

    model = Ridge(alpha = alpha, normalize = True, random_state = 9)
    model.fit(X_train, y_train)

    y_fit_train = model.predict(X_train)
    y_fit_test = model.predict(X_test)

    rmse_train = math.sqrt(mean_squared_error(y_train, y_fit_train))

    rmse_test = math.sqrt(mean_squared_error(y_test, y_fit_test))


    return rmse_train, rmse_test
