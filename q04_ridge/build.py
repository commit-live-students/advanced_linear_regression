# %load q04_ridge/build.py
# Default imports
from sklearn.linear_model import Ridge
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from greyatomlib.advanced_linear_regression.q01_load_data.build import load_data
from sklearn.linear_model import Ridge
# We have already loaded the data for you
data_set, X_train, X_test, y_train, y_test = load_data('data/house_prices_multivariate.csv')

np.random.seed(9)


# Write your solution here
def ridge(alpha=0.01):
    ridge_model=Ridge(alpha=0.01,normalize=True, random_state=9)
    a = ridge_model.fit(X_train, y_train)
    x_pred = ridge_model.predict(X_train)
    x = np.sqrt(mean_squared_error(y_train,x_pred))
    y_pred = ridge_model.predict(X_test)
    y = np.sqrt(mean_squared_error(y_test,y_pred))
    return x, y, a

ridge()

