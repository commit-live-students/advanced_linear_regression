# %load q03_polynomial/build.py
# Default imports
from greyatomlib.advanced_linear_regression.q01_load_data.build import load_data
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression

# We have already loaded the data for you
data_set, X_train, X_test, y_train, y_test = load_data('data/house_prices_multivariate.csv',0.33,9)

def polynomial(power=5, random_state=9):
    cols = data_set.corr().nlargest(5, 'SalePrice')['SalePrice'].index
    model=make_pipeline(PolynomialFeatures(degree=power, include_bias=False), LinearRegression())
    model.fit(X_train[cols[1:]], y_train)
    return model




