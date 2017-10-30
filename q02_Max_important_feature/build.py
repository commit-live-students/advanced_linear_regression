# Default imports
from greyatomlib.advanced_linear_regression.q01_load_data.build import load_data
import numpy as np
# We have already loaded the data for you
data_set, X_train, X_test, y_train, y_test = load_data('data/house_prices_multivariate.csv')
# Write your code here
# def Max_important_feature(data_set,target_variable='SalePrice',n=4):
#     k = n
#     cols = data_set.corr().nlargest(k, target_variable)[target_variable].index
#     return np.array(cols[1:5])
def Max_important_feature(data_set,target_variable='SalePrice',n=4):
    features=data_set.iloc[:,:-1]
    target=data_set[target_variable]
    four=(features.corrwith(target).sort_values(ascending=False).head(n))
    var=np.array(four.index)
    return var
