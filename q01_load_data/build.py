# %load q01_load_data/build.py
# Default imports
import pandas as pd
from sklearn.model_selection import train_test_split


# Write your solution here
def load_data(path,test_size=0.33,Random_state=9):
    df = pd.read_csv(path,index_col=0)
    Y=df.iloc[:,33]
    x=df.iloc[:,:34]
    X_train, X_test, y_train, y_test = train_test_split(x,Y,test_size=test_size, random_state=Random_state)
    return df,X_train, X_test, y_train, y_test
dff,X_train, X_test, y_train, y_test = load_data(path='data/house_prices_multivariate.csv')

X_train.shape


