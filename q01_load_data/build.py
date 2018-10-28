# %load q01_load_data/build.py
# Default imports
import pandas as pd
from sklearn.model_selection import train_test_split


# Write your solution here
def load_data(path,test_size=0.33,Random_state=9):
    data = pd.read_csv(path)
    X_train,X_test,y_train,y_test = train_test_split(data.iloc[:,:-1],data.iloc[:,-1],random_state=Random_state,test_size=test_size)
    
    return data, X_train,X_test,y_train,y_test



