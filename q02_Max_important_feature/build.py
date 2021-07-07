# %load q02_Max_important_feature/build.py
# Default imports
from greyatomlib.advanced_linear_regression.q01_load_data.build import load_data

# We have already loaded the data for you
data_set, X_train, X_test, y_train, y_test = load_data('data/house_prices_multivariate.csv')
target_variable = data_set['SalePrice']
#print(target_variable)

def Max_important_feature(data,ytest='SalePrice',n=4):

    Correlation = []

    corr=data.corr(method='pearson')['SalePrice']
    corr = corr.sort_values(ascending=False)

    Correlation = corr.index.tolist()

    return Correlation[1:5]
Max_important_feature(data_set, y_test, 4)

# Write your code here


'OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea'

