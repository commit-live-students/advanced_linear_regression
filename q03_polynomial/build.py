# Default imports
from greyatomlib.advanced_linear_regression.q01_load_data.build import load_data
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression

# We have already loaded the data for you
data_set, X_train, X_test, y_train, y_test = load_data('data/house_prices_multivariate.csv')

def polynomial(power = 5,rs = 9):
    poly_reg =PolynomialFeatures(degree = power, include_bias=False)
    X_poly = poly_reg.fit_transform(X_train[['OverallQual','GrLivArea','GarageArea', 'GarageCars']])
    lin_reg  = LinearRegression()
    model = lin_reg.fit(X_poly,y_train)
    #print (model)
    return model
