# Default imports
import pandas as pd
from sklearn.model_selection import train_test_split


path = 'data/house_prices_multivariate.csv'


# Write your solution here
# Write your solution here
def load_data(path, test_size=0.33, rand_state=9):
    data = pd.read_csv(path)
    house_attribs = data.iloc[:,:-1]
    sale_price = data.iloc[:,-1]
    X_train, X_test, y_train, y_test = train_test_split(house_attribs, sale_price, test_size=test_size, \
                                        random_state=rand_state)
    return (data, X_train, X_test, y_train, y_test)
