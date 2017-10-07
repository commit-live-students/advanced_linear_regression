import sys, os
import pandas as pd
sys.path.append(os.path.join(os.path.dirname(os.curdir)))

from unittest import TestCase
from q01_load_data.build import load_data


class TestLoad_data(TestCase):
    def test_load_data(self):
        data, X_train, X_test, y_train, y_test = load_data('data/house_prices_multivariate.csv')

        self.assertTrue(type(data), pd.DataFrame)
        self.assertTrue(len(X_test), 455)



