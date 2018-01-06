# Default imports
from sklearn.linear_model import Ridge
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from greyatomlib.advanced_linear_regression.q01_load_data.build import load_data
np.random.seed(9)

# We have already loaded the data for you
data_set, X_train, X_test, y_train, y_test = load_data('data/house_prices_multivariate.csv')


# Write your solution here
from sklearn.linear_model import Ridge
def ridge(alpha=0.01):
    ridge_model = Ridge(alpha=alpha, normalize=True, random_state=9)
    m = ridge_model.fit(X_train, y_train)
    y_pred_train = m.predict(X_train)
    y_pred_test = m.predict(X_test)
    RMSE_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
    RMSE_test = np.sqrt(mean_squared_error(y_test,y_pred_test))
    return RMSE_train, RMSE_test#, m

# rmsetrain, rmsetest, m = ridge()
# print "RMSE Train: {0}:  dtype: {1}".format(rmsetrain,type(rmsetrain))
# print "RMSE Test: {0}:  dtype: {1}".format(rmsetest,type(rmsetest))
# print m
# Too many values to unpack error, removing model as return variable fiexed it
# However the question says return model. There is obviously a difference between the question and test cases
