# Default imports
from sklearn.model_selection import cross_val_score
import numpy as np
from greyatomlib.advanced_linear_regression.q01_load_data.build import load_data

np.random.seed(9)
# We have already loaded the data for you
data_set, X_train, X_test, y_train, y_test = load_data('data/house_prices_multivariate.csv')

# Write your solution here

from sklearn.linear_model import Ridge
data_set, X_train, X_test, y_train, y_test = load_data('data/house_prices_multivariate.csv')
model_ridge = Ridge(alpha=0.01,normalize=True,random_state=9)

def cross_validation(model_ridge,X_test,y_train):
    value = cross_val_score(model_ridge,X_train,y_train,cv=5,scoring='mean_squared_error')
    final_score = value.mean()
    return final_score
