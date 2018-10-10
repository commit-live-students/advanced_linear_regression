# %load q03_polynomial/build.py
# Default imports
from greyatomlib.advanced_linear_regression.q01_load_data.build import load_data
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# We have already loaded the data for you
data_set, X_train, X_test, y_train, y_test = load_data('data/house_prices_multivariate.csv')


# Write your solution here
def polynomial(power = 5, random_state = 9):
    # Function to import import numerical feature which have highest corr
    def Max_important_feature(data_set,target_variable = 'SalePrice' , n = 4):
        col = data_set.corr().nlargest(n=(n+1),columns=target_variable)[target_variable].index[1:]
        return list(col)

    def load_data(df,test_size = 0.33, random_state = random_state):
        X = df.iloc[:,:-1]
        y = df['SalePrice']
        X_train, X_test, y_train, y_test = train_test_split(X, y,  random_state=random_state, test_size= test_size)
        return df, X_train, X_test, y_train, y_test

    col = Max_important_feature(data_set,target_variable = 'SalePrice' , n = 4)

    df = data_set[col]
    df['SalePrice'] = data_set['SalePrice']
    df, X_train, X_test, y_train, y_test = load_data(df,test_size = 0.33, random_state = random_state)

    pipeline = make_pipeline(PolynomialFeatures(degree=power,include_bias=False),LinearRegression())
    model = pipeline.fit(X_train , y_train)
    return model

polynomial(power = 5, random_state = 9)










