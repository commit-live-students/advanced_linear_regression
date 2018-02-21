# %load q02_Max_important_feature/build.py
# Default imports
from greyatomlib.advanced_linear_regression.q01_load_data.build import load_data
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# We have already loaded the data for you
data_set, X_train, X_test, y_train, y_test = load_data('data/house_prices_multivariate.csv')


#saleprice correlation matrix
def Max_important_feature(data_set,target_variable='SalePrice',n=4):    
    k = 5 #number of variables for heatmap
    cols = data_set.corr().nlargest(k, 'SalePrice')['SalePrice'].index
    return list(cols)[1:5]


