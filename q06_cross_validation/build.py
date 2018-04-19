# %load q06_cross_validation/build.py
# Default imports
from sklearn.model_selection import cross_val_score
from sklearn import linear_model 
import numpy as np
from greyatomlib.advanced_linear_regression.q01_load_data.build import load_data

# We have already loaded the data for you
data_set, X_train, X_test, y_train, y_test = load_data('data/house_prices_multivariate.csv')
np.random.seed(9)

#reg = Ridge(alpha=a,normalize=True,random_state=n)
reg = linear_model.LinearRegression()

def cross_validation(model,X , y ):
    #kf = KFold(n_splits=5, random_state=None, shuffle=False)
    scores = cross_val_score(model, X ,y , scoring='neg_mean_squared_error', cv=5)
    return scores.mean()

cross_validation(reg,X_train,y_train)



