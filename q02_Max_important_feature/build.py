# Default imports
from greyatomlib.advanced_linear_regression.q01_load_data.build import load_data

# We have already loaded the data for you
data_set, X_train, X_test, y_train, y_test = load_data('data/house_prices_multivariate.csv')


# Write your code here
def Max_important_feature(data_set, target_variable="SalePrice", n=4):
    # cor = data_set.corr()[target_variable]
    # return cor.sort_values(ascending=False)[1:n + 1].index.values
    listcorr=[]
    listcol=[]
    for col in data_set.columns:
#         print('{}  {}'.format(col,df['SalePrice'].corr(df[col])))
        listcorr.append(data_set[target_variable].corr(data_set[col]))
        listcol.append(col)
    iowacorr=dict(zip(listcol, listcorr))
    return sorted(iowacorr, key=iowacorr.get, reverse=True)[1:n+1]
