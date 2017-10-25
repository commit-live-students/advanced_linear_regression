# Default imports
from greyatomlib.advanced_linear_regression.q01_load_data.build import load_data
import numpy as np
# We have already loaded the data for you
data_set, X_train, X_test, y_train, y_test = load_data('data/house_prices_multivariate.csv')

# Write your code here
def Max_important_feature(data_set, target_variable = "SalePrice", n = 4):
    corr_all = data_set.corr().abs()
    corr_with_target = corr_all[:][target_variable]
    top_corr = corr_with_target.nlargest(n+1)
    top_corr_array = top_corr.keys()
    top_corr_array = np.array(top_corr_array)
    top_corr_array = top_corr_array[1:]
    #print type(top_corr_array)
    #print top_corr_array
    return top_corr_array
