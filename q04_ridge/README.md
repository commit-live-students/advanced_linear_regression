# Create a Ridge regressor model and measure the accuracy of your linear model.

So far so good,
Feeling expert on building polynomial basis functions huh...

Now ,as you know implementing polynomial basis function increases model complexity and the problem of overfitting arises.
Doing this assignment would provide you how to create and apply Ridge regressor method of 
regularization in action.

We have already loaded the data with the help of function `load_data`.

## Write a function `ridge` that :
- Should return a model with implementation of L2 (Ridge)  Regularization.
- Should be able to fit model on X_train, y_train.

Note :- 
- While using the `Ridge` function imported from the library `sklearn.linear_model` set the default value of `normalize` as True and also the random state should be set as 9.

### Parameters:


| Parameter | dtype | argument type | default value | description |
| :---: | :---: | :---: | :---: | :---: |
| alpha | Numeric Number | compulsory | 0.01 | learning_rate |

### Returns:

| Return | dtype | description |
| --- | --- | --- | 
| RMSE for train | float | Rmse for Ridge regressor |
| RMSE for test | float | Rmse for Ridge regressor |
| Model |  | Model for Ridge linear regression |
