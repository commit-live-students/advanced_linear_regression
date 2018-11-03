# %load q03_polynomial/build.py
# Default imports
from greyatomlib.advanced_linear_regression.q01_load_data.build import load_data
from sklearn.preprocessing import PolynomialFeatures,StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.pipeline import Pipeline
import numpy as np
# We have already loaded the data for you
data_set, X_train, X_test, y_train, y_test = load_data('data/house_prices_multivariate.csv')


# Write your solution here
def polynomial(power=5,Random_state=9):
    #model = make_pipeline(PolynomialFeatures(include_bias=False), LinearRegression())
    poly = PolynomialFeatures(include_bias=False)
    linear_reg = LinearRegression()
    X_train = data_set.loc[:,['OverallQual','GrLivArea','GarageCars','GarageArea']]
    y_train = data_set.loc[:,'SalePrice']
    X_train = poly.fit(X_train)
    y_train = poly.fit(y_train)
    linear_reg.fit(X_train, y_train)

    return linear_reg







