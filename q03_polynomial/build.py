
# %load q03_polynomial/build.py
# Default imports
import pandas as pd
from sklearn.model_selection import train_test_split
from greyatomlib.advanced_linear_regression.q01_load_data.build import load_data
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
# We have already loaded the data for you
data_set, X_train, X_test, y_train, y_test = load_data('data/house_prices_multivariate.csv')


# Write your solution here
def polynomial(power=5,random_state=9):
    model = make_pipeline(PolynomialFeatures(5, include_bias=False),LinearRegression())
    df = pd.read_csv('data/house_prices_multivariate.csv')
    X = df.iloc[:,:-1]
    y = df['SalePrice']
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.33,random_state=9)
    X_train1=X_train[['OverallQual','GrLivArea','GarageCars','GarageArea']]
    model.fit(X_train1,y_train)
    return model


