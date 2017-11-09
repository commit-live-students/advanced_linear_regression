# Default imports
from greyatomlib.advanced_linear_regression.q01_load_data.build import load_data
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression

# We have already loaded the data for you
data_set, X_train, X_test, y_train, y_test = load_data('data/house_prices_multivariate.csv')


# Write your solution here
def polynomial(p=5, random_state=9):
    features = ['OverallQual','GrLivArea','GarageCars','GarageArea']
    X = data_set[features]
    y = data_set.SalePrice
    pipeline = make_pipeline(PolynomialFeatures(degree=p),
                             LinearRegression())
    return pipeline.fit(X, y)
