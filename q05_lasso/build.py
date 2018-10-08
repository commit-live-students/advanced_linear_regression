# %load q05_lasso/build.py
# Default imports
from sklearn.linear_model import Lasso
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from greyatomlib.advanced_linear_regression.q01_load_data.build import load_data
from math import sqrt
# We have already loaded the data for you
data_set, X_train, X_test, y_train, y_test = load_data('data/house_prices_multivariate.csv',test_size=0.33)

np.random.seed(9)


# Write your solution here
def lasso(alpha=0.01):
    features= X_train
    las = Lasso(alpha=0.01,random_state=9,normalize=True)
    las.fit(features, y_train)
    ytest=las.predict(X_test)
    ytrain=las.predict(features)
    rmse1= sqrt(mean_squared_error(y_train,ytrain))
    rmse2= sqrt(mean_squared_error(y_test,ytest))
    return rmse1,rmse2


