# %load q02_Max_important_feature/build.py
# Default imports
from greyatomlib.advanced_linear_regression.q01_load_data.build import load_data

# We have already loaded the data for you
data_set, X_train, X_test, y_train, y_test = load_data('data/house_prices_multivariate.csv')

def Max_important_feature(data_set, target_variable='SalePrice', n=4):
    corr = data_set.corr()
    target_corr = corr.loc[:,target_variable][:-1]
    top4 = target_corr.sort_values(ascending = False)#[:n]#.index.tolist()
    final = top4[0:n].index.tolist()
    return final
Max_important_feature(data_set, target_variable='SalePrice', n=4)


