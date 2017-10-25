# %load q02_Max_important_feature/build.py
# Default imports
from greyatomlib.advanced_linear_regression.q01_load_data.build import load_data
from inspect import getargspec
import numpy as np

# We have already loaded the data for you
data_set, X_train, X_test, y_train, y_test = load_data('data/house_prices_multivariate.csv')


# Write your code here
def Max_important_feature(data_set, target_variable='SalePrice',n=4):
    correlation =  (data_set.corr()[target_variable]).nlargest(n+1)
    return np.array(correlation.drop(target_variable).index)
