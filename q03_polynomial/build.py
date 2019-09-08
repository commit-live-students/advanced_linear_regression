# %load q03_polynomial/build.py
# Default imports
import numpy as np
from greyatomlib.advanced_linear_regression.q01_load_data.build import load_data
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from inspect import getargspec

# We have already loaded the data for you
data_set, X_train, X_test, y_train, y_test = load_data('data/house_prices_multivariate.csv')

def polynomial(power = 5, Random_State = 9):
    data = X_train[['OverallQual','GrLivArea','GarageCars','GarageArea']]
    poly_model = make_pipeline(PolynomialFeatures(power, include_bias=False), LinearRegression())
    poly_model.fit(data[:], y_train)
    #y_pred = poly_model.predict(x_test[:, np.newaxis])
    return poly_model

#polynomial(power=5, Random_State=9)
args = getargspec(polynomial)
print len(args[0])
print args[3]
model = polynomial()
prediction = model.predict(np.array([4, 5, 6, 7]).reshape(1, -1))
print np.round_(prediction,2) # This should be equal to [32740.9]
print np.array([np.round_(32740.9,2)])


