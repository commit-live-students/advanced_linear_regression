# Default imports
from greyatomlib.advanced_linear_regression.q01_load_data.build import load_data
import numpy as np

# We have already loaded the data for you
data_set, X_train, X_test, y_train, y_test = load_data('data/house_prices_multivariate.csv')


# Write your code here
def Max_important_feature(data_set,target_variable= "SalePrice",n = 4):
    c = data_set.corr()
    Correlation = c[c[target_variable] < 1][target_variable].sort_values(ascending=False).head(n).index.values
    
    return Correlation
