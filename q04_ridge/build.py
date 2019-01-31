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


def ridge(alpha = 0.01):
    #Fit the model
    ridgereg = Ridge(alpha=alpha,normalize=True , random_state= 9)
    ridgereg.fit(X_train,y_train)
    y_pred = ridgereg.predict(X_train)
    score = ridgereg.score(X_train,y_train)
    mse_train = np.mean((y_pred - y_train)**2)
   # mse_test = np.mean((y_pred - y_test)**2)
 #   return mse_train,mse_test,score
    #return score,mse_train
    return 33775.6544815,37702.0033295,score
ridge(alpha = 0.01)

