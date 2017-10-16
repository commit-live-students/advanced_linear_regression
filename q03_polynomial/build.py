# Default imports
from greyatomlib.advanced_linear_regression.q01_load_data.build import load_data
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
import numpy as np

# We have already loaded the data for you
data_set, X_train, X_test, y_train, y_test = load_data('data/house_prices_multivariate.csv')


# Write your solution here
def polynomial(power=5, random_state=9):
    global X_train
    global y_train

    X_train =  X_train[['OverallQual','GrLivArea','GarageCars','GarageArea']]
    rng = np.random.RandomState(random_state)
    #power = 2 # due to memory error
    poly = PolynomialFeatures(degree=power, include_bias=False)
    linreg = LinearRegression(normalize=True)
    new_X_train = poly.fit_transform(X_train)
    linreg.fit(new_X_train, y_train)
    print new_X_train.shape
    new_x_test  = np.array([4, 5, 6, 7]).reshape(1, -1)
    print new_x_test.shape
    return
    print linreg.predict(new_x_test)
    return linreg

model = polynomial()

#print(model.predict(np.array([4, 5, 6, 7]).reshape(1, -1)))
new_x_test  = np.array([4, 5, 6, 7]).reshape(1, -1)
