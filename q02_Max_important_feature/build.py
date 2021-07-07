# Default imports
from greyatomlib.advanced_linear_regression.q01_load_data.build import load_data

# We have already loaded the data for you
data_set, X_train, X_test, y_train, y_test = load_data('data/house_prices_multivariate.csv')


# Write your code here
def Max_important_feature(data_set, target_varilable='SalePrice', n=4):
    import numpy as np
    d = {}
    for col in data_set.columns.tolist():
        d[col] = data_set[col].corr(data_set[target_varilable])
    del d[target_varilable]
    return np.array(sorted(d, key=d.get, reverse=True)[:n], dtype=object)
