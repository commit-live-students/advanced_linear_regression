# %load q01_load_data/build.py
# Default imports
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def load_data(path, testsize=0.33, rs=9):
    df = pd.read_csv(path)
    X = df.iloc[:,:-1]
    y = df.iloc[:,-1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=rs, test_size=testsize)
    return df, X_train, X_test, y_train, y_test




