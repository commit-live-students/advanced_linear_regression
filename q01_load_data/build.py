# Default imports
import pandas as pd
from sklearn.model_selection import train_test_split


path = 'data/house_prices_multivariate.csv'

def load_data(a,b=0.33,c=9):
    #print (b,c)
    df = pd.read_csv(a)
    X = df.iloc[:,:-1]
    y = df.iloc[:,-1]
    X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=b,random_state=c)
    return df,X_train,X_test,y_train,y_test
