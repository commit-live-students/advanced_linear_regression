# Default imports
from greyatomlib.advanced_linear_regression.q01_load_data.build import load_data
import numpy as np 
# We have already loaded the data for you
data_set, X_train, X_test, y_train, y_test = load_data('data/house_prices_multivariate.csv')


# Write your code here

def Max_important_feature(data_set,target_variable,n=4):
    corr = data_set.corr()
    np.fill_diagonal(corr.values, -2)
    corr = corr[target_variable].sort_values(ascending=False)
    return np.array(corr.index[:n])
