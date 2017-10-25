# Default imports
from greyatomlib.advanced_linear_regression.q01_load_data.build import load_data
import numpy as np
# We have already loaded the data for you
data_set, X_train, X_test, y_train, y_test = load_data('data/house_prices_multivariate.csv')


# Write your code here

def Max_important_feature(data_set,target_variable='SalePrice',n=4):
    features=data_set.iloc[:,:-1]
    target=data_set['SalePrice']
    #print(features.head(5))
    #print(target.head(5))
    four= features.corrwith(target).sort_values(ascending=False).head(4)
    var=np.array(four.index)
    return var
