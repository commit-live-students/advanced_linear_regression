from sklearn.linear_model import Ridge
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from greyatomlib.advanced_linear_regression.q01_load_data.build import load_data

data_set, X_train, X_test, y_train, y_test = load_data('data/house_prices_multivariate.csv')

np.random.seed(9)

def ridge(al=0.01):
    
    ridge_model = Ridge(alpha=al, normalize=True, random_state=9)
    ridge_model.fit(X_train, y_train)
    
    y_pred_train = ridge_model.predict(X_train)
    y_pred_test = ridge_model.predict(X_test)
    
    rmse_train = mean_squared_error(y_train, y_pred_train)**0.5
    rmse_test = mean_squared_error(y_test, y_pred_test)**0.5
    
    return rmse_train, rmse_test, ridge_model

ridge(0.01)




