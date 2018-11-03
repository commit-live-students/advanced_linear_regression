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
def lasso(alpha=0.01):
    model = Lasso(alpha=alpha, random_state=9, normalize=True)
    model.fit(X_train,y_train)
    y_train_pred = model.predict(X_train)
    y_test_pred= model.predict(X_test)
    rmse_train = mean_squared_error(y_train,y_train_pred)**.5
    rmse_test = mean_squared_error(y_test,y_test_pred)**.5
    return float(rmse_train), float(rmse_test)


#rmse_train, rmse_test = lasso()
#print(type(rmse_train))
#print(type(rmse_test))
#print(rmse_train)
#print(rmse_test)


