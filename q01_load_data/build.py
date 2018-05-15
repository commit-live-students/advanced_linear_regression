# Default imports
import pandas as pd
from sklearn.model_selection import train_test_split


path = 'data/house_prices_multivariate.csv'
test_size = .33
random_state = 9

def load_data(path,test_size = .33, random_state= 9):
    df = pd.read_csv(path)
    y = df.iloc[:,-1]
    X = df.loc[:, df.columns != 'SalePrice']
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_state, test_size= test_size)
    return df, X_train, X_test, y_train, y_test
