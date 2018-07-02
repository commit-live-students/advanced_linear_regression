# %load q02_Max_important_feature/build.py
# Default imports
from greyatomlib.advanced_linear_regression.q01_load_data.build import load_data
import pandas as pd
# We have already loaded the data for you
data_set, X_train, X_test, y_train, y_test = load_data('data/house_prices_multivariate.csv')


# Write your code here
def Max_important_feature(data_set,target_variable='SalePrice',n=4):
    result=(data_set.corr()[target_variable])
    result=pd.DataFrame(result)
    result.sort_values(by=target_variable,ascending=False,inplace=True)
    result=result[1:]
    return list(result[:4].index)

    




