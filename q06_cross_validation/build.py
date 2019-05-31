# %load q06_cross_validation/build.py
# Default imports
from sklearn.model_selection import cross_val_score

from sklearn.linear_model import Ridge
import numpy as np
from greyatomlib.advanced_linear_regression.q01_load_data.build import load_data

# We have already loaded the data for you
data_set, X_train, X_test, y_train, y_test = load_data('data/house_prices_multivariate.csv')
np.random.seed(9)

X = data_set.iloc[:,:-1]
y = data_set['SalePrice']

# Write your solution here
def cross_validation(Model,X,y):
    
    scores = cross_val_score(Model, X, y, scoring='neg_mean_squared_error', cv=5)
    return scores.mean()

cross_validation(Ridge(alpha=0.1),X_train,y_train)
    
    
    
    



