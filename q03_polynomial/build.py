# Default imports
from greyatomlib.advanced_linear_regression.q01_load_data.build import load_data
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
import numpy as np

# We have already loaded the data for you
data_set, X_train, X_test, y_train, y_test = load_data('data/house_prices_multivariate.csv')
def polynomial(power=5,rs = 9):
    np.random.RandomState(rs)
    X = X_train[['OverallQual','GrLivArea','GarageCars','GarageArea']]
    higher_polynomial = make_pipeline(PolynomialFeatures(degree=power),LinearRegression())
    higher_polynomial.fit(X,y_train)
    return higher_polynomial


# Write your solution here
