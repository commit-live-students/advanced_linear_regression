# %load q03_polynomial/build.py
# Default imports
from greyatomlib.advanced_linear_regression.q01_load_data.build import load_data
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd

# We have already loaded the data for you
data_set, X_train, X_test, y_train, y_test = load_data('data/house_prices_multivariate.csv')


# Write your solution here
def polynomial(power = 5,random_state = 9):
    
    poly = PolynomialFeatures(degree=5,include_bias=False)
    X = data_set.iloc[:,:-1]

    y = data_set['SalePrice']
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.33,random_state=9)
    x = X_train.loc[:,['OverallQual','GrLivArea','GarageCars','GarageArea']]

    poly_model = make_pipeline(poly,LinearRegression())
    poly_model.fit(x,y_train)
    return poly_model
polynomial(power = 5,random_state = 9)




