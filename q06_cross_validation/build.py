# Default imports
from sklearn.model_selection import cross_val_score
import numpy as np
from greyatomlib.advanced_linear_regression.q01_load_data.build import load_data

np.random.seed(9)
# We have already loaded the data for you
data_set, X_train, X_test, y_train, y_test = load_data('data/house_prices_multivariate.csv')

# Write your solution here
def cross_validation(model,x,y):
# Write your solution here
    scores = cross_val_score(model,x,y,scoring="neg_mean_squared_error", cv=5)
    return scores.mean()
