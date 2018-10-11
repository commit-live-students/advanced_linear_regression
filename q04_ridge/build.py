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
    ridge_model = Ridge(alpha=0.01, normalize=True, random_state=9)
    ridge_model.fit(X_train, y_train)
    
    train_pred = ridge_model.predict(X_train)
    train_pred = pd.DataFrame(train_pred, columns=['Ridge_predict'])
    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    
    test_pred = ridge_model.predict(X_test)
    test_pred = pd.DataFrame(test_pred,columns=['Ridge_predict'])
    test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
    
    return train_rmse, test_rmse, ridge_model
       
ridge(0.01)

