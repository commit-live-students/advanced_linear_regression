import numpy as np
from greyatomlib.advanced_linear_regression.q01_load_data.build import load_data
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression

# We have already loaded the data for you
data_set, X_train, X_test, y_train, y_test = load_data('data/house_prices_multivariate.csv')

X_train=X_train[['OverallQual','GrLivArea','GarageCars','GarageArea']]

def polynomial(power = 5,random_state = 9):
    np.random.seed = random_state
    poly_model=PolynomialFeatures(X_train,y_train)
    poly_model=make_pipeline(PolynomialFeatures(power,include_bias=False),LinearRegression())
    return poly_model.fit(X_train,y_train)
