# Default imports
from sklearn.linear_model import Lasso
import pandas as pd
import numpy as np
from math import sqrt
from sklearn.metrics import mean_squared_error
from greyatomlib.advanced_linear_regression.q01_load_data.build import load_data
np.random.seed(9)

# We have already loaded the data for you
data_set, X_train, X_test, y_train, y_test = load_data('data/house_prices_multivariate.csv')

def lasso(alpha=0.01):
    model=Lasso(alpha=alpha,normalize=True,random_state=9)

    model.fit(X_train,y_train)
    x_pred=model.predict(X_train)
    y_pred=model.predict(X_test)
    rms1=sqrt(mean_squared_error(y_train,x_pred))
    rms2=sqrt(mean_squared_error(y_test,y_pred))
    return rms1,rms2

# Write your solution here
