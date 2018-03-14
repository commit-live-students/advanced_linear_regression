# Default imports
from greyatomlib.advanced_linear_regression.q01_load_data.build import load_data

data_set, X_train, X_test, y_train, y_test = load_data('data/house_prices_multivariate.csv')


# Write your code here
def Max_important_feature(data_set, target_variable='SalePrice', n=4):
    cor = data_set.corr()[target_variable]
    return cor.sort_values(ascending=False)[1:n + 1].index.values


