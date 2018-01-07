# %load q01_load_data/build.py
# Default imports
import pandas as pd
from sklearn.model_selection import train_test_split


# Write your solution here
path = "data/house_prices_multivariate.csv"

def load_data(path, test_size=0.33, random_state=9):
    data = pd.read_csv(path) # The entire data set
    # In this dataset, the dep variable is the last column SalePrice
    # Hence, we remove it to form the ind variable set X
    X = data.iloc[:,:-1] # ind
    y = data['SalePrice'] # dep

    # use the train test split method and create the datasets
    df,X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_state, test_size=test_size)
    return df, X_train, X_test, y_train, y_test
    
