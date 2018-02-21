# %load q01_load_data/build.py
# Default imports
import pandas as pd
from sklearn.model_selection import train_test_split
from inspect import getargspec

path = 'data/house_prices_multivariate.csv'
split=0.33

def load_data(path,split=0.33,rs=9):
    data = pd.read_csv(path)
    X = data.iloc[:,:-1]
    y = data['SalePrice']

    # split the data with 50% in each set
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=rs, train_size=0.67)
    return data, X_train, X_test, y_train, y_test

#data, X_train, X_test, y_train, y_test = load_data(path)
#X_test.iloc[5,4]
#print y_train.shape
