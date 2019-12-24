# %load q02_Max_important_feature/build.py
# Default imports
from greyatomlib.advanced_linear_regression.q01_load_data.build import load_data

# We have already loaded the data for you
data_set, X_train, X_test, y_train, y_test = load_data('data/house_prices_multivariate.csv')


# Write your code here
def Max_important_feature(data_set, target_variable='SalePrice', n=4):
    
    value = data_set.corr()['SalePrice'].drop('SalePrice').sort_values(ascending=False)[0:4]
    
    return value.index


Max_important_feature(data_set, target_variable='SalePrice', n=4)



