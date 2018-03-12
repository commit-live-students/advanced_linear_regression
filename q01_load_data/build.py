# %load q01_load_data/build.py
# Default imports
import pandas as pd
from sklearn.model_selection import train_test_split

path = 'data/house_prices_multivariate.csv'

def load_data(path, test_size=0.33, Random_state=9):
    df = pd.read_csv(path)
    X_train, X_test, y_train, y_test = train_test_split(df.iloc[:,0:-1], df.iloc[:,-1], test_size = test_size, random_state = Random_state)
    return df, X_train, X_test, y_train, y_test


