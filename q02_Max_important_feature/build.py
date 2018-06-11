# %load q02_Max_important_feature/build.py
# Default imports
from greyatomlib.advanced_linear_regression.q01_load_data.build import load_data
import pandas as pd
# We have already loaded the data for you
data_set, X_train, X_test, y_train, y_test = load_data('data/house_prices_multivariate.csv')


# Write your code here

def Max_important_feature(data_set,target_variable='SalePrice',n=4):
    
    cor=data_set.corr()
    cor=cor.sort_values(target_variable,ascending=False)
    max_corr=cor.head(n+1)
    return(list(max_corr[target_variable][1:5].index.values))
    




