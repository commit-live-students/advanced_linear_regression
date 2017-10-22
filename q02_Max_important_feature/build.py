# Default imports
from greyatomlib.advanced_linear_regression.q01_load_data.build import load_data

# We have already loaded the data for you
data_set, X_train, X_test, y_train, y_test = load_data('data/house_prices_multivariate.csv')


# Write your code here
def Max_important_feature(data_set,target_variable,n=4):
    target_variable = 'SalePrice'
    correlation = data_set.corr().SalePrice
    c = correlation.sort_values(ascending=False).head(n+1)
    return c[1:]
