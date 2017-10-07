# Find out the max correlated features 

Smooth as butter , 
As the assumptions of linear regression say 
Correlation is bit of problem for linear regression model so here ,we will deal with them.

## Write the function `Max_important_feature` that :
- Calculates the correlation between target variable and all the features.
- Should return Four of the top correlated features.
 
Doing this assignment will help you in obtaining a grasp on what is correlation and how to deal with it.


### Parameters:

| Parameter | dtype | argument type | default value | description |
| --- | --- | --- | --- | --- | 
| data_set | dataframe | compulsory |  | dataframe loaded from load_data() function |
| target_varilable | float | compulsory | | Correlation with the target variable |
| n | int | optional | 4 | no. of correlated output |

### Returns:

| Return | dtype | description |
| --- | --- | --- | 
| Correlation | float | Correlation between variable |

Hint: For this function you can use corr() function.

Let's get started!