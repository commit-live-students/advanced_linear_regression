# Default imports
import pandas as pd
from sklearn.model_selection import train_test_split


path = 'data/house_prices_multivariate.csv'


# Write your solution here
def load_data(path, test_size = 0.33, random_state = 9):
    data = pd.read_csv('data/house_prices_multivariate.csv')
    X_train, X_test, y_train, y_test = train_test_split(data.iloc[:,:-1], data.iloc[:,-1], test_size= test_size, random_state= random_state)
    return data, X_train, X_test, y_train, y_test
