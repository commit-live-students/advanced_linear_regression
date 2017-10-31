
# Default imports
from sklearn.linear_model import Lasso
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

from greyatomlib.advanced_linear_regression.q01_load_data.build import load_data

# We have already loaded the data for you
data_set, X_train, X_test, y_train, y_test = load_data('data/house_prices_multivariate.csv')

np.random.seed(9)

def lasso(alpha=0.01):
    #code to implement l2 and to from
    model = Lasso(alpha=alpha, normalize=True)
    model.fit(X_train, y_train)
    y_predicted_train = model.predict(X_train)
    y_predicted_test = model.predict(X_test)
    train_rmsc = np.sqrt(mean_squared_error(y_train, y_predicted_train))
    test_rmsc = np.sqrt(mean_squared_error(y_test, y_predicted_test))
    return train_rmsc, test_rmsc
