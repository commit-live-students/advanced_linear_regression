# %load q01_load_data/build.py
# Default imports
import pandas as pd
from sklearn.model_selection import train_test_split
#path = 'data/house_prices_multivariate.csv'

# Write your solution here
def load_data(path,Test_size = 0.33,Random_State = 9):
    df = pd.read_csv(path)
    X = df.iloc[:,:-1]
    y = df.iloc[:,-1]
    
    X_train,X_test,y_train,y_test = train_test_split(X,y,random_state  = Random_State,test_size = Test_size )
    
    return df,X_train,X_test,y_train,y_test

#load_data(path)


