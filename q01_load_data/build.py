# %load q01_load_data/build.py
# Default imports
import pandas as pd
from sklearn.model_selection import train_test_split

path = 'data/house_prices_multivariate.csv'

def load_data(path,test_size=0.33,random_state=9):
    data=pd.read_csv(path)
    X = data.iloc[:,:-1]
    y = data['SalePrice']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=9)
    return(data,X_train, X_test, y_train, y_test)




