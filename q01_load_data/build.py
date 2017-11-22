# %load q01_load_data/build.py
# Default imports
import pandas as pd
from sklearn.model_selection import train_test_split

path = 'data/house_prices_multivariate.csv'
def load_data(path,test_size = .33 ,random_state = 9):
    df = pd.read_csv(path)
    rstate = int(random_state)
    tsize = float(test_size)
    X = df.iloc[:,:-1]
    y = df['SalePrice']
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=rstate, train_size=.67)
    return df,X_train, X_test, y_train, y_test
# Write your solution here
