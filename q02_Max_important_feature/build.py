# %load q02_Max_important_feature/build.py
# Default imports
from greyatomlib.advanced_linear_regression.q01_load_data.build import load_data

# We have already loaded the data for you
data_set, X_train, X_test, y_train, y_test = load_data('data/house_prices_multivariate.csv')

def Max_important_feature(data_set,target_variable='SalePrice',n=4):
    df=data_set.corr().unstack().sort_values(ascending=False)
    return df[target_variable][1:5].index
c=Max_important_feature(data_set,target_variable='SalePrice',n=4)
c






