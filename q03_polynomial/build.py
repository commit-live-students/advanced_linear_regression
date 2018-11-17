# %load q03_polynomial/build.py
# Default imports
from greyatomlib.advanced_linear_regression.q01_load_data.build import load_data
from sklearn.preprocessing import PolynomialFeatures,StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.pipeline import Pipeline
import numpy as np
# We have already loaded the data for you



# Write your solution here
def polynomial(power=5,Random_state=9):
    data_set, X_train, X_test, y_train, y_test = load_data('data/house_prices_multivariate.csv',random_state=Random_state)
    arr = np.array(data_set.corr()['SalePrice'].sort_values(ascending=False).index)
    model = make_pipeline(PolynomialFeatures(degree = power, include_bias = False), LinearRegression())
    
    model.fit(X_train.loc[:, arr[1:4+1]],y_train)
    return model








