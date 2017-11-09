# Default imports
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

path = 'data/house_prices_multivariate.csv'

def load_data(path,tst_size=0.33,n=9):
    randomstate = np.random.RandomState(n)
    df = pd.read_csv(path)
    X = df.iloc[:,:-1]
    y = df.iloc[:,-1]
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=tst_size,random_state = randomstate)
    return df, X_train, X_test, y_train, y_test
