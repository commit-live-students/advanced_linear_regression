# %load q02_Max_important_feature/build.py
# Default imports
from greyatomlib.advanced_linear_regression.q01_load_data.build import load_data

# We have already loaded the data for you
data_set, X_train, X_test, y_train, y_test = load_data('data/house_prices_multivariate.csv')


# Write your code here
def Max_important_feature(data_set, target_variable='SalePrice', n=4):
    correlation_data = data_set.corr().nlargest(n+1, target_variable)[target_variable].index
    #Deleting the target variable itself since it has highest correlation
    correlation_data = correlation_data.delete(0)
    correlation_col_list = []
    for col in correlation_data:
        correlation_col_list.append(col)
    return correlation_col_list
    
Max_important_feature(data_set)


