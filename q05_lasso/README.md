## Create a Lasso regressor model and measure the accuracy of your linear model.

Your job is to improvise L1 (lasso)  Regularization on linear model created previously. 


This assignment will help you how to create and apply Lasso regressor method of 
regularization in action and also provide a way to implement feature selection which comes handy 
often times.
 
## Write a function `lasso` that :
- Should return a model with implementation of L1 (Lasso)  Regularization.
- Should be able to fit model on X_train, y_train.

Note :-
- While using the Lasso function imported from the library `sklearn.linear_model` set the default value of normalize as True and also the random state should be set as 9.

### Parameters:

| Parameter | dtype | argument type | default value | description |
| :---: | :---: | :---: | :---: | :---: |
| alpha | Numeric Number | optional | 0.01 | learning_rate |

### Returns:

| Return | dtype | description |
| --- | --- | --- | 
| RMSE for train | float | Rmse for Lasso regressor |
| RMSE for test | float | Rmse for Lasso regressor |
