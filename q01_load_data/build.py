# Default imports
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np
random_state = np.random.RandomState(9)
path = 'data/house_prices_multivariate.csv'
# Write your solution here
def load_data(path,test_size,random_state):
    df = pd.read_csv(path)
    X = df.iloc[:,:-1]
    y = df.iloc[:,-1]
    lr = LinearRegression()
    lr.fit(X,y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
    return df,X_train,X_test,y_train,y_test
