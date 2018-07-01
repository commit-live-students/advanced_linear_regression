# %load q04_ridge/build.py
# Default imports
from sklearn.linear_model import Ridge
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

from math import sqrt
from greyatomlib.advanced_linear_regression.q01_load_data.build import load_data

# We have already loaded the data for you
data_set, X_train, X_test, y_train, y_test = load_data('data/house_prices_multivariate.csv')

np.random.seed(9)

alpha1=0.01
# Write your solution here
def ridge(alpha = 0.01):
    clf = Ridge(alpha=alpha,normalize = True,random_state = 9)
    clf.fit(X_train,y_train)
    y_Predict = clf.predict(X_test)
    rmse_Predict = np.sqrt(mean_squared_error(y_test,y_Predict))
    print(rmse_Predict)
    y_Train_Predict = clf.predict(X_train)
    rmse_Train_Predict = np.sqrt(mean_squared_error(y_train,y_Train_Predict))
    print(rmse_Train_Predict)
    return rmse_Train_Predict,rmse_Predict,clf
ridge(0.01)


