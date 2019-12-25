# %load q04_ridge/build.py
# Default imports
from sklearn.linear_model import Ridge
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from greyatomlib.advanced_linear_regression.q01_load_data.build import load_data

# We have already loaded the data for you
data_set, X_train, X_test, y_train, y_test = load_data('data/house_prices_multivariate.csv')

p=np.random.seed(9)
def ridge(alpha=0.01):
    ridge_model=Ridge(alpha=0.01, normalize=True, random_state=9)
    ridge_model.fit(X_train, y_train)
    y_pred = ridge_model.predict(X_test)
    y_trainp=ridge_model.predict(X_train)
    rmsetest=np.sqrt(mean_squared_error(y_test, y_pred))
    rmsetrain=np.sqrt(mean_squared_error(y_train, y_trainp))
    return(rmsetrain,rmsetest,ridge_model)




