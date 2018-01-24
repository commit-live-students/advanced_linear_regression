# Default imports
from greyatomlib.advanced_linear_regression.q01_load_data.build import load_data
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression

# We have already loaded the data for you
data_set, X_train, X_test, y_train, y_test = load_data('data/house_prices_multivariate.csv')

# Write your solution here

import numpy as np
def Max_important_feature (data_set,target_variable = 'SalePrice',n=4):
    corr = data_set.corr()[target_variable]
    corr = corr[corr != corr.loc[target_variable]]
    return list(  abs(corr).sort_values(ascending= False)[:n].index)

def polynomial  (power= 5,random_state = 9):
    #features = list(Max_important_feature(data_set))
    features =['OverallQual','GrLivArea','GarageCars','GarageArea']
    poly_model = make_pipeline(PolynomialFeatures(power, include_bias=False),
                           LinearRegression())
    poly_model.fit(X_train[features],y_train )
    return poly_model

#print( polynomial().predict(np.array([4, 5, 6, 7]).reshape(1, -1)))
