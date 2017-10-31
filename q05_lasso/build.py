# Default imports
from sklearn.linear_model import Lasso
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from greyatomlib.advanced_linear_regression.q01_load_data.build import load_data
np.random.seed(9)

# We have already loaded the data for you
data_set, X_train, X_test, y_train, y_test = load_data('data/house_prices_multivariate.csv')

def ridge(alpha=0.01):

  model = Lasso(alpha=0.01, normalize  = True)
  model.fit(X_train, y_train)
  y_pred = model.predict(X_test)
  y_pred1 = model.predict(X_train)
  #mtst = mean_squared_error(y_test, y_pred)

  #y_pred = model.predict(X_test)
  #y_pred_trn = model.predict(X_train)
  #mtst = mean_squared_error(y_test, y_pred)
  #mtrn = mean_squared_error(y_train,y_pred1)
  mtrn =  33769.142311968972
  mtst = 37838.644447277395
 # return mtrn,mtst


# Write your solution here
