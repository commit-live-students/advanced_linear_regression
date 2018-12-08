# %load q03_polynomial/build.py
# Default imports
from greyatomlib.advanced_linear_regression.q01_load_data.build import load_data
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression

# We have already loaded the data for you
data_set, X_train, X_test, y_train, y_test = load_data('data/house_prices_multivariate.csv')
X_train = X_train[['OverallQual','GrLivArea','GarageCars','GarageArea']]
X_test = X_test[['OverallQual','GrLivArea','GarageCars','GarageArea']]

# Write your solution here
def polynomial(power=5,Random_state=9):
    # PolynomialFeatures (prepreprocessing)
    poly = PolynomialFeatures(degree=power,include_bias=False)
    poly_x = poly.fit_transform(X_train)
    regressor=LinearRegression()
    model = regressor.fit(poly_x,y_train)
    return model
    



