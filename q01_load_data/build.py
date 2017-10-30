# Default imports
import pandas as pd
from sklearn.model_selection import train_test_split


path = 'data/house_prices_multivariate.csv'

def load_data(path, test_size=0.33,random_state=9):
    df = pd.read_csv('data/house_prices_multivariate.csv')
    X = df.iloc[:,:-1]
    y = df['SalePrice']


    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=9, train_size=1-test_size)
    return df,X_train, X_test, y_train, y_test
