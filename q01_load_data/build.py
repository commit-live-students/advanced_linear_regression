# %load q01_load_data/build.py
# Default imports
import pandas as pd
from sklearn.model_selection import train_test_split
from greyatomlib.linear_regression.q01_load_data.build import load_data
import numpy as np

path = 'data/house_prices_multivariate.csv'
# Write your solution here
def load_data(path, test_size=0.33,random_state=9):
    df = pd.read_csv('data/house_prices_multivariate.csv')
    X = df.iloc[:,:-1]
    y = df['SalePrice']

    # split the data with 50% in each set
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=9, train_size=1-test_size)
    return df,X_train, X_test, y_train, y_test
