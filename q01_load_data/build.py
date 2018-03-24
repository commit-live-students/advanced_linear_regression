# Default imports
import pandas as pd
from sklearn.model_selection import train_test_split


path = 'data/house_prices_multivariate.csv'


# Write your solution here
def load_data(path , test_size = 0.33 , Randomstate=9 ):

    data=pd.read_csv(path)
    X = data.iloc[:,:-1]
    y = data.iloc[:,-1]
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=test_size,random_state=Randomstate)
    return data,X_train,X_test,y_train,y_test


#print(load_data(path,1/3,9))

# data=pd.read_csv(path)
# print(data.head())
