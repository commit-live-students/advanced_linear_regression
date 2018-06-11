# %load q01_load_data/build.py
# Default imports
import pandas as pd
from sklearn.model_selection import train_test_split


# Write your solution here

#path,test_size,RandomState

def load_data(path,test_size=0.33,RandomSate=9):
    
    
    df=pd.read_csv(path)
    X_train,X_test,y_train,y_test= train_test_split(df.iloc[:,:34],df['SalePrice'],test_size=test_size, random_state=RandomSate)
    
    return(df,X_train,X_test,y_train,y_test)
    



