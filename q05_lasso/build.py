# Default imports
from sklearn.linear_model import Lasso
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from greyatomlib.advanced_linear_regression.q01_load_data.build import load_data
np.random.seed(9)

# We have already loaded the data for you
data_set, X_train, X_test, y_train, y_test = load_data('data/house_prices_multivariate.csv')


# Write your solution here
def lasso(alpha=0.01):
    lasso_model=Lasso(alpha=alpha,normalize=True, random_state=9)
    y_pred_train = lasso_model.fit(X_train, y_train).predict(X_train)
    y_pred_test =  lasso_model.predict(X_test)
    return float(np.sqrt(mean_squared_error(y_train,y_pred_train))),float(np.sqrt(mean_squared_error(y_test,y_pred_test))) 
