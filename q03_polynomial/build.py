# %load q03_polynomial/build.py
#Default imports 
from greyatomlib.advanced_linear_regression.q01_load_data.build import load_data  
from sklearn.preprocessing import PolynomialFeatures 
from sklearn.pipeline import make_pipeline 
from sklearn.linear_model import LinearRegression 
import matplotlib.pyplot as plt 
import numpy as np

def polynomial(power=5,random_state=9):
    data_set, X_train, X_test, y_train, y_test = load_data('data/house_prices_multivariate.csv') 
    cols=['OverallQual','GrLivArea','GarageCars','GarageArea']
    poly_model = make_pipeline(PolynomialFeatures(5,include_bias=False),LinearRegression())
    poly_model.fit(X_train[cols], y_train.reshape(-1,1))
    ypred = poly_model.predict(np.array([4, 5, 6, 7]).reshape(1, -1))
    return(poly_model)
polynomial(power=5,random_state=9)



