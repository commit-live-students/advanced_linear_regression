# %load q03_polynomial/build.py
# Default imports
from greyatomlib.advanced_linear_regression.q01_load_data.build import load_data
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression

# We have already loaded the data for you
data_set, X_train, X_test, y_train, y_test = load_data('data/house_prices_multivariate.csv')

features_list = ['OverallQual','GrLivArea','GarageCars','GarageArea']
# Write your solution here
def polynomial(power=5, random_state=9):
    
    poly = PolynomialFeatures(degree = power, include_bias = False)
    
    poly_model = make_pipeline(poly, LinearRegression())
    poly_model.fit(X_train.loc[:, features_list], y_train.loc[:])
    return poly_model
   




