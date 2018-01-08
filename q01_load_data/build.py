# Default imports
import pandas as pd
from sklearn.model_selection import train_test_split


path = 'data/house_prices_multivariate.csv'



def load_data(path,test_size=0.33,random_state=9):
    dataframe = pd.read_csv(path)
    X = dataframe.iloc[:, :-1]
    y = dataframe.iloc[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size,random_state=random_state)
    return dataframe,X_train,X_test,y_train,y_test
