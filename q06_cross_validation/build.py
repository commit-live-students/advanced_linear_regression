# Default imports
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.linear_model import Ridge
from greyatomlib.advanced_linear_regression.q01_load_data.build import load_data

np.random.seed(9)
# We have already loaded the data for you
data_set, X_train, X_test, y_train, y_test = load_data('data/house_prices_multivariate.csv')
model = Ridge(alpha=0.01,normalize=True,random_state=9)

def cross_validation(model,X_test,y_train):
    value = cross_val_score(model,X_train,y_train,cv=5,scoring='mean_squared_error')
    finalscore = value.mean()

    return finalscore
