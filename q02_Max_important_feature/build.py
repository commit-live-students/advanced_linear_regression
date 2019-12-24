# %load q02_Max_important_feature/build.py
# Default imports
from greyatomlib.advanced_linear_regression.q01_load_data.build import load_data

# We have already loaded the data for you
data_set, X_train, X_test, y_train, y_test = load_data('data/house_prices_multivariate.csv')


# Write your code here
# data_set.corr()
def Max_important_feature(data_set,target_variable='SalePrice',n=4):
    return data_set.drop('SalePrice', axis=1).apply(lambda x: x.corr(data_set.SalePrice)).abs().sort_values(ascending=False).head(n).index

#Call to the function -
Max_important_feature(data_set,'SalePrice',4)




