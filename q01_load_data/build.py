# Default imports
import pandas as pd
from sklearn.model_selection import train_test_split


path = 'data/house_prices_multivariate.csv'


# Write your solution here
def load_data(path, test_size=.33, random_state=9):
    df = pd.read_csv(path)
    X_train, X_test = train_test_split(df, test_size=test_size, random_state=random_state)
    y_train = X_train["SalePrice"]
    y_test = X_test["SalePrice"]
    X_train = X_train.drop("SalePrice", axis=1)
    X_test = X_test.drop("SalePrice", axis=1)
    return df, X_train, X_test, y_train, y_test
