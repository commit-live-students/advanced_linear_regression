# %load q04_ridge/build.py
# Default imports
from sklearn.linear_model import Ridge
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from greyatomlib.advanced_linear_regression.q01_load_data.build import load_data
from math import sqrt

# We have already loaded the data for you
data_set, X_train, X_test, y_train, y_test = load_data('data/house_prices_multivariate.csv')

np.random.seed(9)


# Write your solution here
def ridge(alpha=0.01):
    ridgereg = Ridge(alpha=alpha,normalize=True, max_iter=1e5)
    ridgereg.fit(X_train, y_train)

    y_pred = ridgereg.predict(X_test)
    rms = sqrt(mean_squared_error(y_test, y_pred))

    ytrain = ridgereg.predict(X_train)
    rms1 = sqrt(mean_squared_error(y_train, ytrain))
    return rms1, rms

#print ridge(alpha=0.01)
