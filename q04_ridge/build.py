# %load q04_ridge/build.py
# Default imports
from sklearn.linear_model import Ridge
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from greyatomlib.advanced_linear_regression.q01_load_data.build import load_data

# We have already loaded the data for you
data_set, X_train, X_test, y_train, y_test = load_data('data/house_prices_multivariate.csv')

np.random.seed(9)


# Write your solution here
def ridge(Alpha=0.01):
    
    ridge = Ridge(alpha = Alpha,normalize=True,random_state=9)
    
    #training the model.
    model = ridge.fit(X_train,y_train)
    
    #Predicting on training data and test data.
    Train_Pred = model.predict(X_train)
    Test_Pred  = model.predict(X_test)
    
    #RMSE for train and test data.
    Train_RMSE = np.sqrt(mean_squared_error(Train_Pred,y_train))
    Test_RMSE  = np.sqrt(mean_squared_error(Test_Pred,y_test))
    
    return Train_RMSE,Test_RMSE,model
    



#Call to the function
ridge()


