# Default imports
from sklearn.model_selection import cross_val_score
import numpy as np
from greyatomlib.advanced_linear_regression.q01_load_data.build import load_data

np.random.seed(9)
# We have already loaded the data for you
data_set, X_train, X_test, y_train, y_test = load_data('data/house_prices_multivariate.csv')

# Write your solution here
def cross_validation(Model,X_train,y_train):
    score = cross_val_score(estimator=Model,X=X_train,y=y_train,scoring='neg_mean_squared_error',cv=5)
    return score.mean()
