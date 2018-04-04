from greyatomlib.advanced_linear_regression.q01_load_data.build import load_data
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression

data_set, X_train, X_test, y_train, y_test = load_data('data/house_prices_multivariate.csv')

cor=data_set.corr()['SalePrice']
features=cor.sort_values(ascending=False)[1:5].index.values
poly =PolynomialFeatures(5,include_bias=False)
X_transform = poly.fit_transform(X_train[features])
X_test_new = poly.fit_transform(X_test[features])

def polynomial(power=5,r_state=9):
    lin_regressor = LinearRegression()
    lin_regressor.fit(X_transform,y_train)
    return lin_regressor

y_pred = polynomial(power=5,r_state=9).predict(X_test_new)
print(y_pred.shape)


