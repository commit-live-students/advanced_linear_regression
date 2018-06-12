# %load q05_lasso/build.py
# Default imports
from sklearn.linear_model import Lasso
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from greyatomlib.advanced_linear_regression.q01_load_data.build import load_data

# We have already loaded the data for you
data_set, X_train, X_test, y_train, y_test = load_data('data/house_prices_multivariate.csv')

np.random.seed(9)

def lasso(alpha=0.01):
    
    lassoReg = Lasso(alpha=alpha,normalize=True,random_state=9)
    model=lassoReg.fit(X_train,y_train)
    
    y_predTrain=lassoReg.predict(X_train)
    y_predTest=lassoReg.predict(X_test)
    rmse1=np.sqrt(mean_squared_error(y_predTrain,y_train))
    rmse2=np.sqrt(mean_squared_error(y_predTest,y_test))
    return(rmse1,rmse2)
# Write your solution here



