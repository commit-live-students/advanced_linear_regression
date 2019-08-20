# Default imports
from sklearn.model_selection import cross_val_score
import numpy as np
from greyatomlib.advanced_linear_regression.q01_load_data.build import load_data

np.random.seed(9)
# We have already loaded the data for you
data_set, X_train, X_test, y_train, y_test = load_data('data/house_prices_multivariate.csv')

# Write your solution here
def cross_validation(model, X, y):
    cv_results = cross_val_score(model, X=X_train,y=y_train, cv=5, scoring='neg_mean_squared_error')
    return cv_results.mean()
