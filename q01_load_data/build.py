# %load q01_load_data/build.py
# Default imports
import pandas as pd
from sklearn.model_selection import train_test_split
path = 'data/house_prices_multivariate.csv'
test_size = 0.33
random_state = 9
# Write your solution here
def load_data(path,test_size=0.33,random_state=9):
    df = pd.read_csv(path)
    len_cols = len(df.columns)
    X = df.iloc[:,:-1]
    y = df.iloc[:,-1]
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=test_size,random_state=random_state)  
    return df, X_train, X_test, y_train, y_test 

load_data(path,test_size=0.33,random_state=9)    




