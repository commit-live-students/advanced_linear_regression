# %load q02_Max_important_feature/build.py
# Default imports
from greyatomlib.advanced_linear_regression.q01_load_data.build import load_data
import pandas as pd
# We have already loaded the data for you
data_set, X_train, X_test, y_train, y_test = load_data('data/house_prices_multivariate.csv')

def Max_important_feature(data_set, target_variable, n=4):

    df = data_set.corr()

    df2 = df.reindex(df[target_variable].abs().sort_values().index)
    df3 = df2[target_variable]
    df4 = df3.tail(n+1)
    df4.drop(df4.index[n], inplace=True)
 #   names = df4.columns.values
    names = list(df4.index.values)
    return names









#
#
# Max_important_feature(data_set, 'SalePrice', 4)
