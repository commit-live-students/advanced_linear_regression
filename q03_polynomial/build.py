# %load q03_polynomial/build.py
# Default imports
from greyatomlib.advanced_linear_regression.q01_load_data.build import load_data
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
import numpy as np
from sklearn.model_selection import train_test_split

# We have already loaded the data for you
data_set, X_train, X_test, y_train, y_test = load_data('data/house_prices_multivariate.csv')


# Write your solution here
def polynomial(power=5,Random_state=9):
    linear_pipe = make_pipeline(PolynomialFeatures(degree=power,include_bias=False), LinearRegression())

    cols = data_set.corr()['SalePrice'].drop('SalePrice').sort_values(ascending = False)[0:4].index
    X = data_set[cols]
#     print(X.shape)
    y = data_set['SalePrice']
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=9, test_size = 0.33)

    linear_pipe.fit(X_train,y_train)
    print(linear_pipe.predict(np.array([4, 5, 6, 7]).reshape(1,-1)))
    return linear_pipe
polynomial(power=5,Random_state=9)




