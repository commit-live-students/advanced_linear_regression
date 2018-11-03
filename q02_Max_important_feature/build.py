# %load q02_Max_important_feature/build.py
# Default imports
from greyatomlib.advanced_linear_regression.q01_load_data.build import load_data
import pandas as pd
# We have already loaded the data for you
data_set, X_train, X_test, y_train, y_test = load_data('data/house_prices_multivariate.csv')


# Write your code here
def Max_important_feature(date_set,target_variable='SalePrice',n=4):
    #correlation = 0
    corrDict = {}
    for column in data_set.columns:
        if column != target_variable:
            #print(column)
            correlation = data_set[column].corr(data_set[target_variable])
            #print(correlation)
            corrDict[column] = correlation
            #print(corrDf  
    #print(corrDict)
    data = pd.DataFrame.from_dict(corrDict, orient='index')
    topMost = list(data.sort_values(0,ascending=False).head(n).index)
    return topMost



