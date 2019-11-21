# %load q01_load_data/build.py
# Default imports
import pandas as pd
from sklearn.model_selection import train_test_split


path = 'data/house_prices_multivariate.csv'


# Write your solution here




#os.getcwd()
#df = pd.read_csv('spam.csv',encoding = 'Latin - I')
#df1 = df.copy()
#cols_to_be_dropped = list(df)(-3:)
##df = df.drop(cols_to_be_dropped,axis = 1)
#df.head()
#df.rename(columns=('v1' : 'status','v2' : 'message')
#laptops = pd.read_csv('laptops.csv' , encoding = 'Latin - I')
#list(laptops)


#def clean_Col(string):
 #   string = string.strip()
  #  string = string.replace(' ','')
   # string = string.replace(' ',)
# Default imports
import pandas as pd
from sklearn.model_selection import train_test_split


# Write your solution here

def load_data(path, test_size=0.33, random_state=9):
    df = pd.read_csv(path)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return df, X_train, X_test, y_train, y_test

df = pd.read_csv(path)
#df.head()
df = pd.read_csv(path)
X = df
y = df['SalePrice']
#    X_train, X_test, y_train, y_test = train_test_split(df, test_size = test_size, random_state = Random_state)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.33, random_state = 9)

X_train.shape

