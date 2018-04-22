# %load q03_polynomial/build.py
# Default imports
from greyatomlib.advanced_linear_regression.q01_load_data.build import load_data
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression

# We have already loaded the data for you
data_set, X_train, X_test, y_train, y_test = load_data('data/house_prices_multivariate.csv')

def polynomial(power=5,Random_state=9):
    df = make_pipeline(PolynomialFeatures(degree=5,include_bias=False),LinearRegression())
    return(df.fit(X_train.loc[:,['OverallQual','GrLivArea','GarageCars','GarageArea']],y_train))

#polynomial(power=5,Random_state=9)
# Write your solution here



