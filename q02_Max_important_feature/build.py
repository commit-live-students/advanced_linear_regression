# %load q02_Max_important_feature/build.py
# Default imports
import numpy as np
from greyatomlib.advanced_linear_regression.q01_load_data.build import load_data

# We have already loaded the data for you
data_set, X_train, X_test, y_train, y_test = load_data('data/house_prices_multivariate.csv')
target_variable='SalePrice'

d={}
# Write your code here
def Max_important_feature (data_set,target_variable='SalePrice',n=4):
    #return data_set[data_set.columns[1:]].corr()['SalePrice'][:-1]
    for col in data_set.columns.tolist():
        d[col]=data_set[col].corr(data_set[target_variable])
    del d[target_variable]
    return np.array(sorted(d,key=d.get,reverse=True)[:n],dtype=object)

#Max_important_feature (data_set,'SalePrice',4)
