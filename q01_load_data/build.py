import pandas as pd
from sklearn.model_selection import train_test_split
def load_data(path,test_size=0.33,randomState=9):
    data = pd.read_csv(path)
    X = data.iloc[:,:-1]
    y = data['SalePrice']
    X_train, X_test, y_train, y_test = train_test_split(X, y,random_state=randomState, test_size=test_size)
    return data,X_train, X_test, y_train, y_test
