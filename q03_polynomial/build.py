# Default imports
from greyatomlib.advanced_linear_regression.q01_load_data.build import load_data
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
import numpy as np

# We have already loaded the data for you
data_set, X_train, X_test, y_train, y_test = load_data('data/house_prices_multivariate.csv')
def polynomial(power=5,random=9):
    #model = polynomial()
    poly_model = make_pipeline(PolynomialFeatures(5),LinearRegression())
    poly_model.fit_transform(X_train[:, None])
    return poly_model.predict(np.array([4, 5, 6, 7]).reshape(1, -1))



# Write your solution here
