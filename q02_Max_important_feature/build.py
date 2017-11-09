# Default imports
from greyatomlib.advanced_linear_regression.q01_load_data.build import load_data

# We have already loaded the data for you
data_set, X_train, X_test, y_train, y_test = load_data('data/house_prices_multivariate.csv')


def Max_important_feature(data,target_variable,n=4):
    data_corr =  data_set.corr()

    return data_corr.sort_values(by='SalePrice', ascending=False)['SalePrice'].iloc[1:5].index.values
