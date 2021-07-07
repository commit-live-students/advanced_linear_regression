# %load q03_polynomial/build.py
# Default imports
from greyatomlib.advanced_linear_regression.q01_load_data.build import load_data
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression

# We have already loaded the data for you
data_set, X_train, X_test, y_train, y_test = load_data('data/house_prices_multivariate.csv')


# Write your solution here
def polynomial(pow=5, r_state=9):
    
    cor=data_set.corr()['SalePrice']
    features=cor.sort_values(ascending=False)[1:5].index.values
    poly_model=make_pipeline(PolynomialFeatures(degree=pow,include_bias=False,interaction_only=False),LinearRegression(copy_X=True,fit_intercept=True,n_jobs=1,normalize=False))
    poly_model.fit(X_train[features], y_train)
    return poly_model


