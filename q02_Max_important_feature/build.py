import pandas as pd
import numpy as np
from greyatomlib.advanced_linear_regression.q01_load_data.build import load_data

data_set, X_train, X_test, y_train, y_test = load_data('data/house_prices_multivariate.csv')

def Max_important_feature(data_set, target_variable='SalePrice', n=4):
    
    cols = data_set.corr().nlargest(n+1, target_variable)[target_variable].drop([target_variable]).index
    
    cm = data_set[cols].corr()
    
    return cm

Max_important_feature(data_set, 'SalePrice')



