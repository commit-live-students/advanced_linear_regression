# Default imports
from greyatomlib.advanced_linear_regression.q01_load_data.build import load_data

# We have already loaded the data for you
data_set, X_train, X_test, y_train, y_test = load_data('data/house_prices_multivariate.csv')
import numpy as np


def Max_important_feature(data_set,target_variable = 'SalePrice', n = 4 ):
    cor = (data_set.corr()[target_variable]).nlargest(n+1)
    return np.array(cor.drop(target_variable).index)



# Write your code here
