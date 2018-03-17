# Default imports
from sklearn.linear_model import Ridge
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from greyatomlib.advanced_linear_regression.q01_load_data.build import load_data
np.random.seed(9)

# We have already loaded the data for you
data_set, X_train, X_test, y_train, y_test = load_data('data/house_prices_multivariate.csv')

from math import sqrt
def ridge(alpha = 0.01):
    ridgereg = Ridge(alpha=alpha,normalize=True, random_state=9)
    model = ridgereg.fit(X_train,y_train)
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    rms_train = sqrt(mean_squared_error(y_train, y_pred_train))
    rms_test = sqrt(mean_squared_error(y_test, y_pred_test))
    return rms_train,rms_test
