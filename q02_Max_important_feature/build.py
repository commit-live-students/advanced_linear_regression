# %load q02_Max_important_feature/build.py
# Default imports
import pandas as pd
from sklearn.model_selection import train_test_split
from greyatomlib.advanced_linear_regression.q01_load_data.build import load_data

# We have already loaded the data for you
data_set, X_train, X_test, y_train, y_test = load_data('data/house_prices_multivariate.csv')


# Write your code here
def Max_important_feature(path, test_s=0.33, Random_state = 4):
    df = pd.read_csv(path)
    X = df.iloc[:,:-1]
    y = df.iloc[:,-1]
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = test_s, random_state= Random_state)
    return df, X_train, X_test, y_train, y_test





