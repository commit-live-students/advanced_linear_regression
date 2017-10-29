# %load q02_Max_important_feature/build.py
# Default imports
from greyatomlib.advanced_linear_regression.q01_load_data.build import load_data
import numpy as np
# We have already loaded the data for you
data_set, X_train, X_test, y_train, y_test = load_data('data/house_prices_multivariate.csv')

def Max_important_feature(df,target_variable,n=4):
    x=df.corr().iloc[:-1,]
    y=x[target_variable].sort_values().tail(n)
# Write your code here
    return y.index
