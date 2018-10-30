# %load q06_cross_validation/build.py
# Default imports
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
import numpy as np
from greyatomlib.advanced_linear_regression.q01_load_data.build import load_data
from greyatomlib.advanced_linear_regression.q04_ridge.build import ridge

# We have already loaded the data for you
data_set, X_train, X_test, y_train, y_test = load_data('data/house_prices_multivariate.csv')
np.random.seed(9)

# Function to run cross validation for model 
def cross_validation(Model, X , y ):
    cvs = cross_val_score(Model,X,y,cv=5, scoring='neg_mean_squared_error')
    return cvs.mean()

cross_validation(Ridge(alpha=0.1), X_train, y_train)




