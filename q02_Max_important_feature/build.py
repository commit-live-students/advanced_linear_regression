# %load q02_Max_important_feature/build.py
# Default imports
from greyatomlib.advanced_linear_regression.q01_load_data.build import load_data

# We have already loaded the data for you
data_set, X_train, X_test, y_train, y_test = load_data('data/house_prices_multivariate.csv')
target_variable = 'SalePrice'
n = 4
# Write your code here
def Max_important_feature(data_set, target_variable = 'SalePrice', n = 4):
    CorrelationAll = data_set.corr()
    CorrelationAll = CorrelationAll[target_variable].sort_values(ascending=False).drop(labels=[target_variable])
    Correlation = CorrelationAll.nlargest(n=4).index.values
    return Correlation



