# %load q01_load_data/build.py
# Default imports
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
# from numpy.random import RandomState

path = ('data/house_prices_multivariate.csv')

# Write your solution here

def load_data (path,test_size=0.33,random_state=9):
    data = pd.read_csv('data/house_prices_multivariate.csv')
    X=data.iloc[:,:-1]
    Y=data['SalePrice']
    X_train, X_test, y_train, y_test = train_test_split( X, Y,random_state=9,train_size=1-test_size)
    print(y_test.shape,X_train.shape,y_train.iloc[4])
    return data,X_train, X_test, y_train, y_test


load_data (path,0.33,9)
