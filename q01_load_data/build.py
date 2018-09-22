# %load q01_load_data/build.py
# Default imports
import pandas as pd
from sklearn.model_selection import train_test_split


# Write your solution here
def load_data(path, size=0.33,state=9):
    df = pd.read_csv(path)
    #print(df.head(5))
    y = df['SalePrice']
    X = df.drop(['SalePrice'],axis=1)
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=size,random_state=state)
    return df, X_train,X_test,y_train,y_test


