# Default imports
from sklearn.linear_model import Ridge
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from greyatomlib.advanced_linear_regression.q01_load_data.build import load_data
np.random.seed(9)
from math import sqrt 
# We have already loaded the data for you
data_set, X_train, X_test, y_train, y_test = load_data('data/house_prices_multivariate.csv')


# Write your solution here
def ridge(alp=0.01):
    ridge_model=Ridge(alpha=alp, random_state=9,normalize=True)
    ridge_model.fit(X_train, y_train)
    # evaluate the model on the second set of data
    y_pred = ridge_model.predict(X_test)
    y2 = ridge_model.predict(X_train)

    return sqrt(mean_squared_error(y_train, y2)),sqrt(mean_squared_error(y_test, y_pred))
