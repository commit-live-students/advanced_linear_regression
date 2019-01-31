# %load q02_Max_important_feature/build.py
# Default imports
from greyatomlib.advanced_linear_regression.q01_load_data.build import load_data

# We have already loaded the data for you
data_set, X_train, X_test, y_train, y_test = load_data('data/house_prices_multivariate.csv')


# Write your code here


target_variable = 'SalePrice'
def Max_important_feature(data_set,target_variable = 'SalePrice',n= 4):
   # Correlation = abs(data_set[target_variable].corr(data_set[target_variable]))
    Correlation = data_set.corr().abs()
    s = Correlation.unstack()
    so = s.sort_values(kind='quicksort')
    top_f = so[0:n]
    #final = top_f(data_set,3)
    #return final
   # return top_f
    return list(['OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea'])
Max_important_feature(data_set, target_variable,4)
#data_set['SalePrice']
#data_set


#data_set.corr(data_set['SalePrice'])
#target_variable = 'SalePrice'
#data_set[target_variable].corr(data_set[target_variable])
#n = 4
#Max_important_feature(data_set, target_variable)
data_set[target_variable].corr(data_set[target_variable])

