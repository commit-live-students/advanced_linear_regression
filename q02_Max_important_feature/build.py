# Default imports
from greyatomlib.advanced_linear_regression.q01_load_data.build import load_data

# We have already loaded the data for you
data_set, X_train, X_test, y_train, y_test = load_data('data/house_prices_multivariate.csv')


# Write your code here
def Max_important_feature(data_set, target_variable = 'SalePrice', n = 4):
    c = data_set.iloc[:,:-1].corrwith(data_set[target_variable])
    df = c.abs().sort_values(ascending=False).head(n)
    return list(df.index)
