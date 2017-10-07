# Load the data and split it into test and train data set 

Hello budding Data scientist , of all the process we have done of loading and 
splitting of data separately .
Why not do it in single step.  


## Write a function `load_data` that :
- Load the data into a dataframe of csv format.
- Split the data into train and test data set.
- Set the split ratio as 33 percent.

### Parameters:

| Parameter | dtype | argument type | default value | description |
| --- | --- | --- | --- | --- | 
| path | string | compulsory |  | path to the file csv |
| test_size | float | optional | | test split |
| Random state | Integer | compulsory | | Random state |

### Returns:

| Return | dtype | description |
| --- | --- | --- | 
| df | Dataframe for Data; CSV acceptable by sklearn | X_train |
| X_train | Dataframe for training, testing; CSV acceptable by sklearn | X_train |
| X_test | Dataframe for training, testing; CSV acceptable by sklearn | X_test |
| y_train | Dataframe for training, testing; CSV acceptable by sklearn | y_train |
| y_test | Dataframe for training, testing; CSV acceptable by sklearn | y_test |