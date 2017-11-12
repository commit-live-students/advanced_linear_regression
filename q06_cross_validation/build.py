# Default imports
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
import numpy as np
from greyatomlib.advanced_linear_regression.q01_load_data.build import load_data

np.random.seed(9)
# We have already loaded the data for you
data_set, X_train, X_test, y_train, y_test = load_data('data/house_prices_multivariate.csv')

# Write your solution here
def cross_validation(model, X_train, y_train):
    model.fit(X_train, y_train)
    errors = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
    print errors.mean()
    return errors.mean()
