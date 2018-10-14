# %load q02_Max_important_feature/build.py
# Default imports
from greyatomlib.advanced_linear_regression.q01_load_data.build import load_data

# We have already loaded the data for you
data_set, X_train, X_test, y_train, y_test = load_data('data/house_prices_multivariate.csv')


# Write your code here
def Max_important_feature(data_set, a, n=5):
    corr_values = data_set.corr(method='pearson')[a]
    b = corr_values.nlargest(n).index.tolist()
    b.remove('SalePrice')
    return b
Max_important_feature(data_set,a = 'SalePrice', n=5)
