# %load q03_polynomial/build.py
# Default imports
from greyatomlib.advanced_linear_regression.q01_load_data.build import load_data
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
import numpy as np

# We have already loaded the data for you
data_set, X_train, X_test, y_train, y_test = load_data('data/house_prices_multivariate.csv')


# Write your solution here
def polynomial(power=5,random_state = 9):
    #print(X_train)
    X_trainData = X_train[['OverallQual','GrLivArea','GarageCars','GarageArea']]
    poly_model = make_pipeline(PolynomialFeatures(degree=power, include_bias=False),
                           LinearRegression())
    poly_model.fit(X_trainData, y_train)
    return poly_model
model = polynomial(5,9)


