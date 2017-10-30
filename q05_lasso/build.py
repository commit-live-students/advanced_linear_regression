# Default imports
from sklearn.linear_model import Lasso
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from greyatomlib.advanced_linear_regression.q01_load_data.build import load_data
np.random.seed(9)

# We have already loaded the data for you
data_set, X_train, X_test, y_train, y_test = load_data('data/house_prices_multivariate.csv')


from math import sqrt
# Write your solution here
def lasso(alpha_val=0.01):
    lasso_model=Lasso(alpha=alpha_val,normalize=True, random_state=9)

    # fit the model on one set of data
    lasso_model.fit(X_train, y_train)

    # evaluate the model on the second set of data
    y_pred = lasso_model.predict(X_test)
    # print y_pred
    y2 = lasso_model.predict(X_train)
    rmse2= sqrt(mean_squared_error(y_test, y_pred))
    rmse1= sqrt(mean_squared_error(y_train, y2))

    return rmse1,rmse2
