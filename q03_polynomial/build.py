# %load q03_polynomial/build.py
# Default imports
import numpy as np
from greyatomlib.advanced_linear_regression.q01_load_data.build import load_data
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression

# We have already loaded the data for you
data_set, X_train, X_test, y_train, y_test = load_data('data/house_prices_multivariate.csv')

def polynomial(power=5,RandomState=9):
    df_X=X_train[['OverallQual','GrLivArea','GarageCars','GarageArea']]
    higher_polynomial = make_pipeline(PolynomialFeatures(5,include_bias=False),
                            LinearRegression())
    
    model=higher_polynomial.fit(df_X,y_train)
    return model
    #print(data_set.loc[:,['OverallQual','GrLivArea','GarageCars','GarageArea']])
# Write your solution here



