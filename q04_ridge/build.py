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


# Function to run Ridge regularization 
def ridge(alpha = 0.01):
    #Instantiate and fit the model
    ridge = Ridge(alpha=alpha, normalize=True, random_state=9)
    model = ridge.fit(X_train, y_train)
    
    #Predict y values for test and train
    y_pred = model.predict(X_test)
    y_pred_train = model.predict(X_train)

    #Root mean squared error
    RMSE_test = mean_squared_error(y_test,y_pred)**0.5
    RMSE_train = mean_squared_error(y_train,y_pred_train)**0.5
    
    return RMSE_train, RMSE_test, model

ridge(alpha = 0.01)



