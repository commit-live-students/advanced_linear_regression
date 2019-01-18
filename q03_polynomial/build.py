# %load q03_polynomial/build.py
# Default imports
from greyatomlib.advanced_linear_regression.q01_load_data.build import load_data
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression

# We have already loaded the data for you
data_set, X_train, X_test, y_train, y_test = load_data('data/house_prices_multivariate.csv')


# Write your solution here

def polynomial(power=5, random_state=9):
    data_set, X_train, X_test, y_train, y_test = load_data('data/house_prices_multivariate.csv',random_state=random_state)
    features = ['OverallQual','GrLivArea','GarageCars','GarageArea']
    X_train = X_train[features]

    poly_model = make_pipeline(PolynomialFeatures(degree=power, include_bias=False),
                           LinearRegression())

    poly_model.fit(X_train, y_train)

    return poly_model



