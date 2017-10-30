# Default imports
from greyatomlib.advanced_linear_regression.q01_load_data.build import load_data
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression

# We have already loaded the data for you
data_set, X_train, X_test, y_train, y_test = load_data('data/house_prices_multivariate.csv')
def ploynomial(power=5,random_state=9):
    poly_model = make_pipeline(PolynomialFeatures(power),LinearRegression())
    df1 = X_train[["OverallQual","GrLivArea","GarageCars","GarageArea"]]
    model = poly_model.fit(df1,y_train)
    return poly_model
