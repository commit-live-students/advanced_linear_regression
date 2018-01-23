# Default imports
import pandas as pd
from sklearn.model_selection import train_test_split


path = 'data/house_prices_multivariate.csv'
# Write your solution here
def load_data(file,t_size=0.33,r_state=9):
    df=pd.read_csv(file)
    X=df.iloc[:,:-1]
    y=df.SalePrice
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=t_size,random_state=r_state)
    return df,X_train,X_test,y_train,y_test
