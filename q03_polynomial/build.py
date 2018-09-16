# %load q03_polynomial/build.py
# Default imports
from greyatomlib.advanced_linear_regression.q01_load_data.build import load_data
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np

# We have already loaded the data for you
data_set, X_train, X_test, y_train, y_test = load_data('data/house_prices_multivariate.csv')

def Max_important_feature(data, ytest='SalePrice', n=4):
    Correlation = []
    corr = data.corr(method='pearson')['SalePrice']
    corr = corr.sort_values(ascending=False)
    Correlation = corr.index.tolist() 
    return Correlation[1:5]
    
# Write your solution here
def polynomial(power=5, random_state=9):
    features = []
    features = Max_important_feature(data_set)

    poly_model = make_pipeline(PolynomialFeatures(power, include_bias=False), LinearRegression())
    model = poly_model.fit(X_train.loc[:,features],y_train)

    return model    
    
polynomial(5, 9)

