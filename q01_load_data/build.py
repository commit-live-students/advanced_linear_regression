# Default imports
import pandas as pd
from sklearn.model_selection import train_test_split


path = 'data/house_prices_multivariate.csv'


# Write your solution here
def load_data(path, test_size=0.33, random_state=9):
    data = pd.read_csv(path) # The entire data set
    # In this dataset, the dep variable is the last column SalePrice
    # Hence, we remove it to form the ind variable set X
    X = data.iloc[:,:-1] # ind
    y = data['SalePrice'] # dep

    # use the train test split method and create the datasets
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_state, test_size=test_size)
    return data, X_train, X_test, y_train, y_test

# df,xtr,xte,ytr,yte = load_data(path, 0.33)
# print ("###### Testing #######")
# print "df type: {0}".format(type(df))
# print "X_train type: {0}".format(type(xtr))
# print "X_test type: {0}".format(type(xte))
# print "y_train type: {0}".format(type(ytr))
# print "y_test type: {0}".format(type(yte))
