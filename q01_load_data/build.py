# Default imports
import pandas as pd
from sklearn.model_selection import train_test_split


path = 'data/house_prices_multivariate.csv'


# Write your solution here

def load_data(path,size=0.33,randomstate=9):
    df=pd.read_csv(path)
    print df.shape
    X=df.iloc[:,:-1]
    y=df.loc[:,'SalePrice']
    
    X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=randomstate,test_size=size)
    return df,X_train,X_test,y_train,y_test
