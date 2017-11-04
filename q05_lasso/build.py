# Default imports
from sklearn.linear_model import Lasso
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from greyatomlib.advanced_linear_regression.q01_load_data.build import load_data
np.random.seed(9)

# We have already loaded the data for you
data_set, X_train, X_test, y_train, y_test = load_data('data/house_prices_multivariate.csv')

def lasso(a=0.01):
    model = Lasso(alpha=a,normalize=True,random_state=9)
    model = model.fit(X_train,y_train)
    y_train_pred = model.predict(X_train)
    y_test_predict = model.predict(X_test)

    a= np.sqrt(mean_squared_error(y_train,y_train_pred))
    b = np.sqrt(mean_squared_error(y_test,y_test_predict))

    return a,b
