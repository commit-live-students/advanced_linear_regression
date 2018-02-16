# Default imports
import pandas as pd
from sklearn.model_selection import train_test_split


path = 'data/house_prices_multivariate.csv'

def load_data(path, size = 0.33, rand_state = 9):
    df = pd.read_csv(path)
    y = df.iloc[:,-1]
    X = df.iloc[:,:-1]
    X_train,X_test,y_train,y_test = train_test_split(X, y,
                                                     random_state = rand_state,
                                                     train_size = (1 - size))
    return df, X_train, X_test, y_train, y_test

# Write your solution here
