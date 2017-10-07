## Create a Lasso regressor model and measure the accuracy of your linear model.

Your job is to improvise L2 (Ridge)  Regularization on linear model created previously. 


This assignment will help you how to create and apply Lasso regressor method of 
regularization in action and also provide a way to implement feature selection which comes handy 
often times.
 
## Write a function `lasso` that :
- Should return a model with implementation of L1 (Lasso)  Regularization.
- Should be able to fit model on X_train, y_train.

Note : The random seed and as well as random state should be set as 9. 

#### Parameters:


| Parameter | dtype | argument type | default value | description |
| :---: | :---: | :---: | :---: | :---: |
| alpha | Numeric Number | compulsory | 1 | learning_rate |

#### Returns:

| Return | dtype | description |
| --- | --- | --- | 
| RMSE for train | float | Rmse for Ridge regressor |
| RMSE for test | float | Rmse for Ridge regressor |
| Model |  | Model for Ridge linear regression |
