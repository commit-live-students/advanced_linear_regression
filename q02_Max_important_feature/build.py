# %load q02_Max_important_feature/build.py
# Default imports
from greyatomlib.advanced_linear_regression.q01_load_data.build import load_data

# We have already loaded the data for you
data_set, X_train, X_test, y_train, y_test = load_data('data/house_prices_multivariate.csv')

target_variable = 'SalePrice'

def Max_important_feature(data_set,target_variable = 'SalePrice', n=4):
    k=5
    cols = data_set.corr().nlargest(5,target_variable)[target_variable].index
    return cols[1:n+1]
    #correlation = [i for i in cols[1:n]]
    #return cols

Max_important_feature(data_set,target_variable)



# Write your code here
