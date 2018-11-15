# %load q03_polynomial/build.py
# Default imports
from greyatomlib.advanced_linear_regression.q01_load_data.build import load_data
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression

# We have already loaded the data for you
data_set, X_train, X_test, y_train, y_test = load_data('data/house_prices_multivariate.csv')


# Write your solution here
def polynomial(power=5,random_state=9):
    cols=data_set.corr().drop('SalePrice').nlargest(4,'SalePrice')['SalePrice'].index.tolist()
    
    
    Model = make_pipeline(PolynomialFeatures(power,include_bias=False),LinearRegression(random_state))
    
    Model.fit(X_train[cols], y_train)
    return Model


