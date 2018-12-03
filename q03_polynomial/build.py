# %load q03_polynomial/build.py
# Default imports
from greyatomlib.advanced_linear_regression.q01_load_data.build import load_data
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression

# We have already loaded the data for you
data_set, X_train, X_test, y_train, y_test = load_data('data/house_prices_multivariate.csv')


# Write your solution here
def polynomial(power=5,Random_state=9):
    poly_model = make_pipeline(PolynomialFeatures(power,include_bias=False),
                           LinearRegression())
    return poly_model.fit(X_train[['OverallQual','GrLivArea','GarageCars','GarageArea' ]],y_train)

#Call to the function.
polynomial()
#Just to make use of randomstate function. Not a part of the exercise. 
#And to predict the outcome of random values being passed as dataframe to model.

import numpy as np
import pandas as pd

rng = np.random.RandomState(9).rand(52)
viv = 100*rng
vivek = pd.DataFrame(viv.reshape(-1,4))
ypred = poly_model.predict(vivek)
ypred



