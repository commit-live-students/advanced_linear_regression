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
def ridge(alpha=0.01):
    model = Ridge(alpha=alpha,normalize=True,random_state=9)
    model = model.fit(X_train,y_train)
    train_pred = model.predict(X_train)
    test_pred  = model.predict(X_test)
    rmse1 = sqrt(mean_squared_error(train_pred,y_train))
    rmse2 = sqrt(mean_squared_error(test_pred,y_test))
    return (rmse1,rmse2)

ridge()
