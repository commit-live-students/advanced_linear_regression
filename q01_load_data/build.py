# %load q01_load_data/build.py
# Default imports
from greyatomlib.linear_regression.q01_load_data.build import load_data
from greyatomlib.linear_regression.q02_data_splitter.build import data_splitter
import pandas as pd
from sklearn.model_selection import train_test_split


# Write your solution here

def load_data(path,test_size=0.33,random_state=9):
    df = pd.read_csv(path)
    X = df.iloc[:,:-1]
    y = df['SalePrice']
    X_train, X_test, y_train, y_test =train_test_split(X,y, random_state=random_state, train_size=(1-test_size))
    return df,X_train, X_test, y_train, y_test


