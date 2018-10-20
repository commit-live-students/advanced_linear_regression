# %load q01_load_data/build.py
# Default imports
import pandas as pd
from sklearn.model_selection import train_test_split


# Write your solution here
def load_data(path, ts=0.33, rs=9):
    data = pd.read_csv(path)
    X = data.drop(['SalePrice'], axis=1)
    y = data['SalePrice']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=ts, random_state=rs)
    return data, X_train, X_test, y_train, y_test

    
path = 'data/house_prices_multivariate.csv'
load_data(path)


