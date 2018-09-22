# %load q04_ridge/build.py
# Default imports
from sklearn.linear_model import Ridge
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from greyatomlib.advanced_linear_regression.q01_load_data.build import load_data
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
import math

# We have already loaded the data for you
data_set, X_train, X_test, y_train, y_test = load_data('data/house_prices_multivariate.csv')

np.random.seed(9)

def ridge(alpha=0.01):
    model = Ridge(alpha=alpha, normalize=True, random_state=9)
    model.fit(X_train, y_train)
    predict_train = model.predict(X_train)
    predict_train = pd.DataFrame(predict_train)
    predict_test = model.predict(X_test)
    predict_test = pd.DataFrame(predict_test)
    rmse1 = np.sqrt(mean_squared_error(y_train, predict_train))
    rmse2 = np.sqrt(mean_squared_error(y_test, predict_test))
    rmse1=float('%.7f'%rmse1)
    rmse2=float('%.7f'%rmse2)
    return rmse1,rmse2,model
c=ridge(alpha=0.01)
c



