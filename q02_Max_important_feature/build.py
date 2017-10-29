# Default imports
from greyatomlib.advanced_linear_regression.q01_load_data.build import load_data

# We have already loaded the data for you
data_set, X_train, X_test, y_train, y_test = load_data('data/house_prices_multivariate.csv')

def Max_important_feature(data_set,target_variable,n=4):
    x=data_set.corr().iloc[:-1,]
    y=x[target_variable].sort_values().tail(n)
    return y.index

# Write your code here
