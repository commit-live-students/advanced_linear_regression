# %load q03_polynomial/build.py
# Default imports
from greyatomlib.advanced_linear_regression.q01_load_data.build import load_data
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression

# We have already loaded the data for you
data_set, X_train, X_test, y_train, y_test = load_data('data/house_prices_multivariate.csv')


# Write your solution here
def polynomial(power = 5,Random_state = 9):
    poly_model = make_pipeline(PolynomialFeatures(power,include_bias=False),LinearRegression())
    cols = ['OverallQual','GrLivArea','GarageCars','GarageArea']
    poly_learner=poly_model.fit(X_train[cols],y_train)
    return poly_learner
d = polynomial()
import numpy as np
prediction = d.predict(np.array([4, 5, 6, 7]).reshape(1, -1))
prediction



