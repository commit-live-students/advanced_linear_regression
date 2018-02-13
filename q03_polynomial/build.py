# %load q03_polynomial/build.py
# Default imports
from greyatomlib.advanced_linear_regression.q01_load_data.build import load_data
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from greyatomlib.advanced_linear_regression.q02_Max_important_feature.build import Max_important_feature
import numpy as np
# We have already loaded the data for you
data_set, X_train, X_test, y_train, y_test = load_data('data/house_prices_multivariate.csv')


# Write your solution here
def polynomial(power=5, random_state=9):
    cols = ['OverallQual','GrLivArea','GarageCars','GarageArea']
    df = data_set[cols]
    pipeline = make_pipeline(PolynomialFeatures(degree=power),
                             LinearRegression())
    return pipeline.fit(df, data_set.SalePrice)
