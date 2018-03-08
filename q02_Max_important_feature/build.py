# Default imports
from greyatomlib.advanced_linear_regression.q01_load_data.build import load_data

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# We have already loaded the data for you
data_set, X_train, X_test, y_train, y_test = load_data('data/house_prices_multivariate.csv')
# df = pd.read_csv('data/house_prices_multivariate.csv',index_col=0)

# Write your code here
# def Max_important_feature(data_set=df, target_variable='SalePrice', n=4):
#     #saleprice correlation matrix
#     cols = df.corr().nlargest(n, target_variable)[target_variable].index
#     cm = df[cols].corr()


# Write your code here
def Max_important_feature(data_set, target_variable="SalePrice", n=4):
    cor = data_set.corr()[target_variable]
    return cor.sort_values(ascending=False)[1:n + 1].index.values
