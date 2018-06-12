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

def ridge(alpha=0.01):
    
    ridgeReg = Ridge(alpha=0.01, normalize=True,random_state=9)

    model=ridgeReg.fit(X_train,y_train)

    y_predtrain = ridgeReg.predict(X_train)
    y_predtest= ridgeReg.predict(X_test)
    #calculating mse
    rmse1=(np.sqrt(mean_squared_error(y_train,y_predtrain)))
    rmse2=(np.sqrt(mean_squared_error(y_test,y_predtest)))
    #print (mean_squared_error(y_test,y_predtest))
    return rmse1,rmse2,model




