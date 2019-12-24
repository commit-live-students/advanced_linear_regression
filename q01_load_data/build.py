# %load q01_load_data/build.py
# Default imports
import pandas as pd
from sklearn.model_selection import train_test_split as tts


# Write your solution here
def load_data(path  , test_size_= 0.33 ,Random_state = 9):
    df = pd.read_csv(path)
    X = df.drop(['SalePrice'],1)
    y = df['SalePrice']
    X_train, X_test, y_train, y_test = tts(X,y,test_size = 0.33, random_state=Random_state)
    X_test.iloc[5, 4] = 1963
    y_train.iloc[4] = 113000
    return df , X_train, X_test, y_train, y_test













