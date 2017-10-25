# %load q01_load_data/build.py
# Default imports
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

path = 'data/house_prices_multivariate.csv'
random_state = np.random.RandomState()

# Write your solution here
def load_data(path, test_size = 0.33, random_state = 9):
    df = pd.read_csv(path)
    X = df.iloc[:,:-1]
    y = df['SalePrice']

    # split the data with 33% in each set
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_state, test_size=test_size)
    return (df, X_train, X_test, y_train, y_test)

load_data(path, 0.33, 9)[3]
