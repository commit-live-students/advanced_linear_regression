# %load q04_ridge/build.py
# Default imports
from sklearn.linear_model import Ridge
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from greyatomlib.advanced_linear_regression.q01_load_data.build import load_data

# We have already loaded the data for you
data_set, X_train, X_test, y_train, y_test = load_data('data/house_prices_multivariate.csv')

np.random.seed(9)

# Write your solution here
def ridge(alpha=0.01):
    model = Ridge(alpha=alpha, normalize=True, random_state=9)
    model.fit(X_train, y_train)
    return np.sqrt(mean_squared_error(model.predict(X_train), y_train)), np.sqrt(mean_squared_error(model.predict(X_test), y_test)), model
    
ridge()



