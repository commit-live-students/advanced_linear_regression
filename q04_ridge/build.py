# Default imports
from sklearn.linear_model import Ridge
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from greyatomlib.advanced_linear_regression.q01_load_data.build import load_data
np.random.seed(9)

# We have already loaded the data for you
data_set, X_train, X_test, y_train, y_test = load_data('data/house_prices_multivariate.csv')

def ridge(alpha1=0.01):
    #np.random.seed(9)
    model = Ridge(alpha=alpha1,normalize=True,random_state=9)
    model=model.fit(X_train,y_train)
    ytrain_pred= model.predict(X_train)


    ytest_pred = model.predict(X_test)


    a=np.sqrt(mean_squared_error(y_train,ytrain_pred))
    b=np.sqrt(mean_squared_error(y_test,ytest_pred))

    return a,b
