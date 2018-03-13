# Default imports
import pandas as pd
from sklearn.model_selection import train_test_split

path = 'data/house_prices_multivariate.csv'

def load_data(path1,test_size1=0.33,random_state1=9):
    data= pd.read_csv(path1)
    X= data.iloc[:,:-1]
    y= data['SalePrice']
    X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=test_size1,random_state=random_state1)
    return data,X_train,X_test,y_train,y_test
