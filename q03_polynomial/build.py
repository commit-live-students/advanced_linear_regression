# %load q03_polynomial/build.py
# Default imports
from greyatomlib.advanced_linear_regression.q01_load_data.build import load_data
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression

# We have already loaded the data for you
data_set, X_train, X_test, y_train, y_test = load_data('data/house_prices_multivariate.csv')


# Write your solution here
def polynomial(power = 5, Random_state = 9):
    corre = data_set.corr()['SalePrice']
    featu = corre.sort_values(ascending = False)[1:5].index.values
    poly_model=make_pipeline(PolynomialFeatures(power,include_bias=False),LinearRegression())
    return poly_model.fit(X_train[featu],y_train)
polynomial()



