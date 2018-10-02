# %load q06_cross_validation/build.py
# Default imports
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.linear_model import Ridge,LinearRegression
from greyatomlib.advanced_linear_regression.q01_load_data.build import load_data

# We have already loaded the data for you
data_set, X_train, X_test, y_train, y_test = load_data('data/house_prices_multivariate.csv')
np.random.seed(9)


Model = Ridge(alpha=0.1)
# Write your solution here
def cross_validation(Model, X_train , y_train):
    scores = (cross_val_score(Model, X_train, y_train, scoring = 'neg_mean_squared_error',cv=5)).mean()
    return scores
cross_validation(Model,X_train,y_train)

