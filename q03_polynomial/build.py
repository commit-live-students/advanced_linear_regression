# Default imports
from greyatomlib.advanced_linear_regression.q01_load_data.build import load_data
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression

# We have already loaded the data for you
data_set, X_train, X_test, y_train, y_test = load_data('data/house_prices_multivariate.csv')


# Write your solution here
def polynomial(power=5,rand_state=9):
    X_train_val = X_train[['OverallQual','GrLivArea','GarageCars','GarageArea']]
    poly_regression_model = make_pipeline(PolynomialFeatures(degree=power,include_bias=False), LinearRegression())
    poly_regression_model.fit(X_train_val, y_train)
    return poly_regression_model
