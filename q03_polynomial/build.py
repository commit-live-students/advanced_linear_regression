# Default imports
from greyatomlib.advanced_linear_regression.q01_load_data.build import load_data
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
import numpy as np

# We have already loaded the data for you
data_set, X_train, X_test, y_train, y_test = load_data('data/house_prices_multivariate.csv')
def polynomial(power=5,randomstate = 9):
    #linear_regression = LinearRegression()

    #extractedData = data_set[:,[2,13,23,24]]
   # model = make_pipeline([('poly', PolynomialFeatures(degree=5)), ('linear', LinearRegression(fit_intercept=False))])
    #poly_model = make_pipeline(PolynomialFeatures(5),LinearRegression())
    #model = polynomial()

    linear_regression = LinearRegression()
    model = Pipeline([("polynomial_features", PolynomialFeatures),("linear_regression", linear_regression)])

    #prediction = model.predict(np.array([4, 5, 6, 7]).reshape(1, -1))
    return model


polynomial(power=5,randomstate = 9)



# Write your solution here
