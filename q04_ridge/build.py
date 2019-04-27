# Default imports
from sklearn.linear_model import Ridge
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from greyatomlib.advanced_linear_regression.q01_load_data.build import load_data
from scipy import sqrt

np.random.seed(9)

# We have already loaded the data for you
data_set, X_train, X_test, y_train, y_test = load_data('data/house_prices_multivariate.csv')


# Write your solution here
def ridge(alpha=0.01):
    ridge_model=Ridge(alpha=alpha,normalize=True,random_state=9)

    # fit the model on one set of data
    ridge_model.fit(X_train, y_train)

    # evaluate the model on the second set of data
    y_pred1 = ridge_model.predict(X_test)
    y_pred2 = ridge_model.predict(X_train)
    rmse2 = sqrt(mean_squared_error(y_test, y_pred1))
    rmse1 = sqrt(mean_squared_error(y_train, y_pred2))
    return rmse1,rmse2
