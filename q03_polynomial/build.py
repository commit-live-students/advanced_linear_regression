# %load q03_polynomial/build.py
# Default imports
from greyatomlib.advanced_linear_regression.q01_load_data.build import load_data
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression, Lasso, Ridge
import numpy as np

# We have already loaded the data for you
data_set, X_train, X_test, y_train, y_test = load_data('data/house_prices_multivariate.csv')
np.random.seed(9)

# Write your solution here
def polynomial(power=5, randomState=9):

    higher_polynomial = make_pipeline(PolynomialFeatures(5,include_bias=True),
                            LinearRegression())
    #X = data_set.iloc[:,:-1]
    #y = data_set['SalePrice']
    features = X_train[['OverallQual','GrLivArea','GarageCars','GarageArea']]

    #lasso_model=Lasso(alpha=140, max_iter=100000, random_state=9)
    #liner=LinearRegression()
    return higher_polynomial.fit(features,y_train)
    #y_pred = linear_model.predict(X_test)
    #mean_squared_error(y_test, y_pred)


#model=polynomial()
#print(model.predict(np.array([4, 5, 6, 7]).reshape(1, -1)))
