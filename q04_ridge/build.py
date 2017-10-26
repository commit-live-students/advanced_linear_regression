# %load q04_ridge/build.py
# Default imports
from sklearn.linear_model import Ridge
import pandas as pd
import numpy as np
from math import sqrt
from sklearn.metrics import mean_squared_error
from greyatomlib.advanced_linear_regression.q01_load_data.build import load_data

# We have already loaded the data for you
data_set, X_train, X_test, y_train, y_test = load_data('data/house_prices_multivariate.csv')

np.random.seed(9)


# Write your solution here
def ridge(alpha=0.01):
    ridge_model=Ridge(alpha=alpha, max_iter=100000, random_state=9,normalize=True)
    #X = data_set.iloc[:,:-1]
    #y = data_set['SalePrice']
    #Xupdate=X_train[['OverallQual','GrLivArea','GarageCars','GarageArea']]
    ridge_model.fit(X_train,y_train)
    y_pred_test = ridge_model.predict(X_test)
    y_pred_train = ridge_model.predict(X_train)
    rmse_test=mean_squared_error( y_test,y_pred_test)
    rmse_train=mean_squared_error(y_train,y_pred_train)

    #mse=mean_squared_error(y_train, y_pred_test)
    return (sqrt(rmse_train),sqrt(rmse_test))

# rmse_train,rmse_test,lasso_model=ridge()
# print(type(rmse_train))
# print(type(rmse_test))
# print(rmse_train)
# print(rmse_test)
