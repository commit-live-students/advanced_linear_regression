# %load q02_Max_important_feature/build.py
# Default imports
import numpy as np
import pandas as pd
from greyatomlib.advanced_linear_regression.q01_load_data.build import load_data

# We have already loaded the data for you
data_set, X_train, X_test, y_train, y_test = load_data('data/house_prices_multivariate.csv')


# Function to import import numerical feature which have highest corr
def Max_important_feature(data_set,target_variable = 'SalePrice' , n = 4):
    col = data_set.corr().nlargest(n=(n+1),columns=target_variable)[target_variable].index[1:]
    return list(col)

Max_important_feature(data_set,target_variable = 'SalePrice' , n = 4)






