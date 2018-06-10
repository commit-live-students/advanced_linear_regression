# %load q01_load_data/build.py
# Default imports
import pandas as pd
from sklearn.model_selection import train_test_split


# Write your solution here
def load_data(path,test_size=0.33,randomState = 9):
    df = pd.read_csv(path)
    y_data = df.pop('SalePrice')
    
    X_train,X_test,y_train,y_test = train_test_split(df,y_data,test_size = test_size,random_state = randomState)
    return df,X_train,X_test,y_train,y_test

df,X_train,X_test,y_train,y_test = load_data('data/house_prices_multivariate.csv',0.33,9 )


