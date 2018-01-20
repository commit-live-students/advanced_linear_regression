# %load q05_lasso/build.py
# Default imports
from sklearn.linear_model import Lasso
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from greyatomlib.advanced_linear_regression.q01_load_data.build import load_data

# We have already loaded the data for you
data_set, X_train, X_test, y_train, y_test = load_data('data/house_prices_multivariate.csv')

np.random.seed(9)


# Write your solution here

# Default imports
from sklearn.linear_model import Lasso
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from greyatomlib.advanced_linear_regression.q01_load_data.build import load_data

np.random.seed(9)
data_set, X_train, X_test, y_train, y_test = load_data('data/house_prices_multivariate.csv')


# Write your solution here
def lasso(alpha=0.01):
    lasso = Lasso(alpha=alpha, normalize=True, random_state=9)
    lasso.fit(X_train, y_train)
    predict_train = lasso.predict(X_train)
    predict_train = pd.DataFrame(predict_train, columns=['Lasso_predict'])
    rmse1 = np.sqrt(mean_squared_error(y_train, predict_train))

    predict_test = lasso.predict(X_test)
    predict_test = pd.DataFrame(predict_test, columns=['Lasso_predict'])
    rmse2 = np.sqrt(mean_squared_error(y_test, predict_test))

    return rmse1, rmse2

