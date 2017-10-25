# %load q01_load_data/build.py
# Default imports
import pandas as pd
from sklearn.model_selection import train_test_split


# Write your solution here
def load_data(path,test_size=0.33,RandomState=9):
    df=pd.read_csv("./data/house_prices_multivariate.csv")
    X=df.iloc[:,:-1]
    y=df['SalePrice']
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=RandomState,train_size=0.67)
    return df, X_train, X_test, y_train, y_test


#load_data("./data/house_prices_multivariate.csv")[4]
