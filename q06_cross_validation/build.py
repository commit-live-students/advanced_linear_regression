# %load q06_cross_validation/build.py
# Default imports
from sklearn.model_selection import cross_val_score
import numpy as np
from greyatomlib.advanced_linear_regression.q01_load_data.build import load_data
from sklearn.preprocessing import PolynomialFeatures

# We have already loaded the data for you
data_set, X_train, X_test, y_train, y_test = load_data('data/house_prices_multivariate.csv')
np.random.seed(9)
from sklearn.linear_model import Ridge
# from sklearn.pipeline import make_pipeline
# model = make_pipeline(PolynomialFeatures(15), Lasso(alpha=0.01))


def cross_validation(model,X,y):
    model=Ridge(alpha=0.1)
    scores=cross_val_score(model,X_train,y_train,scoring='neg_mean_squared_error',cv=5)
    return float('%.2f'%scores.mean()
   
c=cross_validation(data_set,X_train,y_train)
c



