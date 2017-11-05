# Default imports
from greyatomlib.advanced_linear_regression.q01_load_data.build import load_data
import numpy as np
import pandas as pd
# We have already loaded the data for you
data_set, X_train, X_test, y_train, y_test = load_data('data/house_prices_multivariate.csv')

def Max_important_feature(data_set, target_varilable, n=4):
    a=data_set.corr()["SalePrice"]
    ser = pd.Series(a.values,a.index)
    a = ser.sort_values(ascending=False)
    b=a[1:5]
    c=np.asarray(b.index)
    return c
