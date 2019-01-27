# %load q03_polynomial/build.py
# Default imports
from greyatomlib.advanced_linear_regression.q01_load_data.build import load_data
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

# We have already loaded the data for you
data_set, X_train, X_test, y_train, y_test = load_data('data/house_prices_multivariate.csv')


# Write your solution here
def polynomial(power=5,random_state=9):
    model = Pipeline([('poly', PolynomialFeatures(degree=power)),
                   ('linear', LinearRegression(fit_intercept=False))])
    model.fit(X_train,y_train)
    return model
import numpy as np
y=np.array([4, 5, 6, 7]).reshape(1, -1)



