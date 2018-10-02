# %load q06_cross_validation/build.py
# Default imports
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.linear_model import Ridge,LinearRegression
from greyatomlib.advanced_linear_regression.q01_load_data.build import load_data

# We have already loaded the data for you
data_set, X_train, X_test, y_train, y_test = load_data('data/house_prices_multivariate.csv')
np.random.seed(9)


# Write your solution here
R_model = Ridge(alpha= 0.1)
model = R_model.fit(X_train, y_train)
def cross_validation(model,X_train, y_train):
    scores = cross_val_score(model,X_train,y_train, scoring='neg_mean_squared_error', cv=5)
    return scores.mean()
cross_validation(model,X_train,y_train)

