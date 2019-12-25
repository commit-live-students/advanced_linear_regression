# %load q05_lasso/build.py
# Default imports
from sklearn.linear_model import Lasso
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from greyatomlib.advanced_linear_regression.q01_load_data.build import load_data
from sklearn.linear_model import Lasso
# We have already loaded the data for you
data_set, X_train, X_test, y_train, y_test = load_data('data/house_prices_multivariate.csv')

np.random.seed(9)


# Write your solution here
def lasso(alpha=0.01):
    lasso_model=Lasso(alpha=0.01,normalize=True, random_state=9)
    lasso_model.fit(X_train, y_train)
    x_pred = lasso_model.predict(X_train)
    x = np.sqrt(mean_squared_error(y_train,x_pred))
    y_pred = lasso_model.predict(X_test)
    y = np.sqrt(mean_squared_error(y_test,y_pred))
    return x, y

lasso()

