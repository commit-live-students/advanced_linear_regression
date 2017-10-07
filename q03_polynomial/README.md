#  To create a  unique polynomial basis function combining top correlated varibles.

Nice work ,keep that spirit.
As we have seen in our lectures we have known what is polynomial basis function and what is it used for.
We can avoid the cause of under-fitting with the help of polynomial basis functions.
With that being said let's implement what we have learned ..

## Write a function `polynomial` that :
- Should return a model with implementation of polynomial basis function.
- All the process should be done with random state set as 9 and power parameter should be set as 5.
- Should be able to load the data with the help of function `load_data`.
- Should be able to extract 'OverallQual','GrLivArea','GarageCars','GarageArea' features
  and fit model on these features.  
 
Doing this assignment will help you learn how to create polynomial basis function and fit it onto linear model.
You can also toy with the parameters and observe the effect on your linear model.

### Parameters:


| Parameter | dtype | argument type | default value | description |
| :---: | :---: | :---: | :---: | :---: |
| power | Numeric Number | compulsory | 5 | power |
| Random state | Numeric Number | compulsory | 9 | random state |

### Returns:

| Return | dtype | description |
| --- | --- | --- | 
| Model |  | Model for polynomial linear regression |

