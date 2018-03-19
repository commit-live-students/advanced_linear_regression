# Default imports
from sklearn.linear_model import Lasso
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from greyatomlib.advanced_linear_regression.q01_load_data.build import load_data
np.random.seed(9)

# We have already loaded the data for you
data_set, X_train, X_test, y_train, y_test = load_data('data/house_prices_multivariate.csv')


# Write your solution here
def lasso(alpha=0.01):
    l1 = Lasso(alpha=alpha,normalize=True,random_state=9)
    l1.fit(X_train,y_train)
    y_pred1 = l1.predict(X_train)
    rmse_train = np.sqrt(mean_squared_error(y_train,y_pred1))
    y_pred2 = l1.predict(X_test)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred2))
    return rmse_train,rmse_test
