# %load q05_lasso/build.py
# Default imports
from sklearn.linear_model import Lasso
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from greyatomlib.advanced_linear_regression.q01_load_data.build import load_data

# We have already loaded the data for you
data_set, X_train, X_test, y_train, y_test = load_data('data/house_prices_multivariate.csv')

np.random.seed(9)


# Write your solution here
def lasso(alpha = 0.01):
    model = Lasso(alpha = alpha,fit_intercept = True,normalize = True,random_state = 9)
    model.fit(X_train,y_train)
    y_Predict = model.predict(X_test)
    y_Test_RMSE = np.sqrt(mean_squared_error(y_test,y_Predict))
    y_Train_Predict = model.predict(X_train)
    y_Train_RMSE = np.sqrt(mean_squared_error(y_train,y_Train_Predict))
    return y_Train_RMSE,y_Test_RMSE
lasso(0.01)

