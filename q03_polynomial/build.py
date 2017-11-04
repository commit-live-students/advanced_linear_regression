# Default imports
from greyatomlib.advanced_linear_regression.q01_load_data.build import load_data
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
import numpy as np

# We have already loaded the data for you
data_set, X_train, X_test, y_train, y_test = load_data('data/house_prices_multivariate.csv')

X = data_set[['OverallQual','GrLivArea','GarageCars','GarageArea']]
y = data_set['SalePrice']

def polynomial(power=5,Random_state = 9):
    np.random.seed(Random_state)
    model = make_pipeline(PolynomialFeatures(power),LinearRegression())
    model.fit(X,y)

    return model
