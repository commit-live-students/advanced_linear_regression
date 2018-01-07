# %load q02_Max_important_feature/build.py
# Default imports
from greyatomlib.advanced_linear_regression.q01_load_data.build import load_data

# We have already loaded the data for you
data_set, X_train, X_test, y_train, y_test = load_data('data/house_prices_multivariate.csv')


# Write your code here
import pandas as pd
def Max_important_feature(data_set, target_variable='SalePrice', n=4):
    cor_ret = data_set.corr().nlargest(n+1, target_variable)[target_variable][1:n+1]
    cor_ret = pd.Series(cor_ret.index.values, index=cor_ret)
    return cor_ret

c = Max_important_feature(data_set, 'SalePrice')
print c
print list(c)
print type(c)



data_set.corr().nlargest(4, 'SalePrice') # Why is sale price also coming? Cannot use this
data_set.corr()['SalePrice'].sort_values(ascending=False)[1:5]


import pandas as pd
e = data_set.corr().nlargest(5, 'SalePrice')['SalePrice'][1:5]
e = pd.Series(e.index.values, index=e)
print e
print list(e)

cor = data_set.corr()['SalePrice']
d = cor.sort_values(ascending=False)[1:5].index.values
list(d)
