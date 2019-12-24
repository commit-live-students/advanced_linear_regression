# %load q04_ridge/build.py
# Default imports
from sklearn.linear_model import Ridge
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from greyatomlib.advanced_linear_regression.q01_load_data.build import load_data

# We have already loaded the data for you
data_set, X_train, X_test, y_train, y_test = load_data('data/house_prices_multivariate.csv')

np.random.seed(9)


# Write your solution here
def ridge(alpha=0.01):
    clf=Ridge(alpha=alpha, normalize=True)
    clf.fit(X_train,y_train)
    y_pred=clf.predict(X_test)
    return mean_squared_error(clf.predict(X_train),y_train)**0.5,mean_squared_error(y_pred,y_test)**0.5,clf
    
    

ridge()
clf=Ridge(alpha=100, normalize=False,random_state=9)
clf.fit(X_train,y_train)
y_pred=clf.predict(X_train)

mean_squared_error(y_pred,y_train) **0.5
from sklearn.linear_model import LinearRegression

clf=LinearRegression()
clf.fit(X_train,y_train)
y_pred=clf.predict(X_train)

mean_squared_error(y_pred,y_train) 
print(y_test.shape,y_pred.shape)
type(y_pred)
y_pred[10]
y_train


