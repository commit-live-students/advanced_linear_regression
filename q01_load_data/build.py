# %load q01_load_data/build.py
# Default imports
import pandas as pd
from sklearn.model_selection import train_test_split
path = 'data/house_prices_multivariate.csv'
Random_state=9
test_size=0.33

# Write your solution here
def load_data(path,test_size = 0.33, Random_state = 9):
    df = pd.read_csv(path)
    X = df.iloc[:,:-1]
    y = df['SalePrice']
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=9, test_size=.33)
    return df, X_train, X_test, y_train, y_test




