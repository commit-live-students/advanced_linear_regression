# Default imports
import pandas as pd
from sklearn.model_selection import train_test_split


path = 'data/house_prices_multivariate.csv'
data_set = pd.read_csv(path)

X = data_set.iloc[:,:-1]
y = data_set['SalePrice']
def load_data(path,test_size=0.33,random_state=9):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size,random_state = random_state)
    return data_set,X_train, X_test, y_train, y_test


# Write your solution here
