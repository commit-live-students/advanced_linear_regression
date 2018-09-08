# %load q02_Max_important_feature/build.py
# Default imports
from greyatomlib.advanced_linear_regression.q01_load_data.build import load_data

# We have already loaded the data for you
data_set, X_train, X_test, y_train, y_test = load_data('data/house_prices_multivariate.csv')

# Write your code here

def Max_important_feature(data_set,y='SalePrice',n=4):
    features=[]

    corr_values = data_set.corr(method='pearson')['SalePrice']
    corr_values = corr_values.abs().sort_values()
    features = corr_values.index.tolist()
    # print (features)
    
    return features[len(features)-n-1:len(features)-1][::-1]

#Max_important_feature(data_set,y,4)



