# %load q06_cross_validation/build.py
# Default imports
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Ridge
import numpy as np
from greyatomlib.advanced_linear_regression.q01_load_data.build import load_data

# We have already loaded the data for you
data_set, X_train, X_test, y_train, y_test = load_data('data/house_prices_multivariate.csv')
np.random.seed(9)

# Write your solution here
model = Ridge(alpha=0.01,normalize=True,random_state=9)
model.fit(X_train,y_train)

def cross_validation(model,X_test,y_test):
    scores = cross_val_score(model,X_test,y_test,scoring='neg_mean_squared_error',cv=5)
    return scores.mean()



