# %load q06_cross_validation/build.py
# Default imports
from sklearn.model_selection import cross_val_score
import numpy as np
from greyatomlib.advanced_linear_regression.q01_load_data.build import load_data
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
data_set, X_train, X_test, y_train, y_test = load_data('data/house_prices_multivariate.csv')
Model=Ridge(alpha=0.1)
# We have already loaded the data for you
def cross_validation(Model, X_train, y_train):
    scores = cross_val_score(Ridge(alpha=0.1),X_train,y_train,scoring='neg_mean_squared_error',cv=5)
    return(scores.mean())

cross_validation(Model, X_train, y_train)



