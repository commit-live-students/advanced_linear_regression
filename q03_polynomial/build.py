# Default imports
from greyatomlib.advanced_linear_regression.q01_load_data.build import load_data
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression,Lasso, Ridge

# We have already loaded the data for you
data_set, X_train, X_test, y_train, y_test = load_data('data/house_prices_multivariate.csv')

def polynomial(power=5, randomState=9):

    higher_polynomial = make_pipeline(PolynomialFeatures(5,include_bias=True),
                            LinearRegression())
    features = X_train[['OverallQual','GrLivArea','GarageCars','GarageArea']]
    return higher_polynomial.fit(features,y_train)
# Write your solution here
