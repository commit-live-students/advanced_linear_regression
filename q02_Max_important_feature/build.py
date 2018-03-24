# Default imports
from greyatomlib.advanced_linear_regression.q01_load_data.build import load_data

# We have already loaded the data for you
data_set, X_train, X_test, y_train, y_test = load_data('data/house_prices_multivariate.csv')


# Write your code here
def Max_important_feature(data_set,target_variable='SalePrice',n=4):
    listcorr = abs(data_set.corr().iloc[-1,:])
    sampledict = {}
    cols=data_set.corr().columns
    it=0
    for j in listcorr:

        sampledict[j] = cols[it]
        it+=1

    listcorr = listcorr.sort_values(ascending=False)

    final=[]
    for i in range(n+1):
        if i==0:
            continue
        final.append(sampledict[listcorr[i]])

    return final




#print(Max_important_feature(data_set,"SalePrice"))
#print(abs(data_set.corr().iloc[-1,:]).sort_values(ascending=False))
