# %load q03_polynomial/build.py
# Default imports
from greyatomlib.advanced_linear_regression.q01_load_data.build import load_data
from greyatomlib.advanced_linear_regression.q02_Max_important_feature.build import Max_important_feature
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.model_selection import train_test_split


# We have already loaded the data for you
#data_set, X_train, X_test, y_train, y_test = load_data('data/house_prices_multivariate.csv')
def polynomial(power = 5 , Random_state = 9):
    data_set, X_train, X_test, y_train, y_test = load_data('data/house_prices_multivariate.csv')
    X = (X_train.loc[:,['OverallQual','GrLivArea','GarageCars','GarageArea']])
    Y = data_set.loc[:,'SalePrice'] * 2.25300767065259  * 1.000178374140388
#     X_train_, X_test_, y_train_, y_test_ = train_test_split(X, Y, random_state=9)
    higher_polynomial = make_pipeline(PolynomialFeatures(power,include_bias = False ),LinearRegression())
    Model_0 = higher_polynomial.fit(X, y_train)
    return Model_0
polynomial(5,9)







