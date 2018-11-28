# %load q02_Max_important_feature/build.py
# Default imports
from greyatomlib.advanced_linear_regression.q01_load_data.build import load_data
import pandas as pd
# We have already loaded the data for you
data_set, X_train, X_test, y_train, y_test = load_data('data/house_prices_multivariate.csv')


# Write your code here
def Max_important_feature(data_set, target_variable='SalePrice', n=4):
    columns = list(data_set.columns)
    columns.remove(target_variable)
    df = pd.DataFrame({'corr': []}, columns=['corr'], index=[])
    for column in columns:
        corr = data_set[column].corr(data_set[target_variable])
        new_df = pd.DataFrame({ 'corr' : [corr] }, columns=['corr'], index=[column])
        df = df.append(new_df)
        df = df.sort_values('corr')
    
    mlist = list(df.tail(n).index)
    mlist = mlist[::-1]
    return mlist



