# %load q04_ridge/build.py
# Default imports
from sklearn.linear_model import Ridge
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from greyatomlib.advanced_linear_regression.q01_load_data.build import load_data

# We have already loaded the data for you

def ridge(alpha = 0.01):
    data_set, X_train, X_test, y_train, y_test = load_data('data/house_prices_multivariate.csv')
    np.random.seed(9)
    ridge_model=Ridge(alpha=alpha, random_state=9, normalize = True)
    ridge = ridge_model.fit(X_train, y_train)
    y_pred_1 = ridge_model.predict(X_train)
    y_pred_2 = ridge_model.predict(X_test)
    rmse_1 = np.sqrt(mean_squared_error(y_pred_1,y_train))
    rmse_2 = np.sqrt(mean_squared_error(y_pred_2, y_test))
    return float(rmse_1),float(rmse_2),ridge
ridge()


# Write your solution here





