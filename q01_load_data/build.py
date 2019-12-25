# %load q01_load_data/build.py
# Default imports
import pandas as pd
from sklearn.model_selection import train_test_split
df = pd.read_csv('data/house_prices_multivariate.csv')
X = df.iloc[:,:-1]
y = df['SalePrice']

# Write your solution here
def load_data(df, test_size = 0.33, random_state = 9):
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=9, test_size=0.33)
    return df, X_train, X_test, y_train, y_test

load_data(df)

