
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression, Lasso, Ridge
import numpy as np
from greyatomlib.advanced_linear_regression.q01_load_data.build import load_data
from greyatomlib.advanced_linear_regression.q04_ridge.build import ridge

data_set, X_train, X_test, y_train, y_test = load_data('data/house_prices_multivariate.csv')
np.random.seed(9)
def cross_validation(Model, X_train, y_train):

    scores = cross_val_score(Model, X_train, y_train, scoring="neg_mean_squared_error", cv=5)
    return scores.mean()



# Write your solution here
