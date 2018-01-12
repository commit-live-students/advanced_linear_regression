# %load q03_polynomial/build.py
# Default imports
from greyatomlib.advanced_linear_regression.q01_load_data.build import load_data
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

# We have already loaded the data for you
data_set, X_train, X_test, y_train, y_test = load_data("data/house_prices_multivariate.csv")


# Write your solution here
import numpy as np
# np.random.seed(9)
def polynomial(power=5, random_state=9):
    # Constructing a pipeline for Linear Reg
    np.random.seed(random_state)
    poly_model = make_pipeline(PolynomialFeatures(power, include_bias=False), LinearRegression())

    # Pick only the columns mentioned
    cols = ['OverallQual','GrLivArea','GarageCars','GarageArea']
#     X_train_cols = X_train[cols]
#     m = poly_model.fit(X_train_cols, y_train)

    # Try running for the whole dataset instead of train only
    # The objective of this exercise is not to split, I think

    data_set_cols = X_train[cols]
    y = y_train
#     print data_set_cols.head()
#     y = data_set['SalePrice']
#     print y.head()
    m = poly_model.fit(data_set_cols, y)

#     Trying with Test dataset now
#     X_test_cols = X_test[cols]
#     poly_model.fit(X_test_cols,y_test)

#     return poly_model, data_set_cols, y
    return m
