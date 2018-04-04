from sklearn.linear_model import Lasso
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from greyatomlib.advanced_linear_regression.q01_load_data.build import load_data

# We have already loaded the data for you
data_set, X_train, X_test, y_train, y_test = load_data('data/house_prices_multivariate.csv')

n=np.random.seed(9)

def lasso(a=0.01):
    model = Lasso(alpha=a,normalize=True,random_state=n)
    model.fit(X_train,y_train)
    y_train_pred=model.predict(X_train)
    rmse1 = np.sqrt(mean_squared_error(y_train,y_train_pred))
    y_pred_test = model.predict(X_test)
    rmse2 = np.sqrt(mean_squared_error(y_test, y_pred_test))
    return rmse1,rmse2
