# Default imports
from greyatomlib.advanced_linear_regression.q01_load_data.build import load_data
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
import numpy as np

# We have already loaded the data for you
data_set, X_train, X_test, y_train, y_test = load_data('data/house_prices_multivariate.csv')

def polynomial(power = 5, rs = 9):

    X = X_train[['OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea']]
    y = y_train

    model = make_pipeline(PolynomialFeatures(power, include_bias = False), LinearRegression())
    model.fit(X, y)

    return model
