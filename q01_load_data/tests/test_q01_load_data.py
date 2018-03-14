import pandas

from unittest import TestCase
from ..build import load_data
from inspect import getfullargspec


class TestLoad_data(TestCase):

    def test_load_data(self):  # Input parameters test
        args = getfullargspec(load_data)
        self.assertEqual(len(args[0]), 3, "Expected argument(s) %d, Given %d" % (3, len(args[0])))

    def test_load_data_default(self):  # Input paramters default
        args = getfullargspec(load_data)
        self.assertEqual(args[3], (0.33, 9), "Expected default values do not match given default values")

    def test_load_data_X_train_type(self):  # Return data types
        data, X_train, X_test, y_train, y_test = load_data('data/house_prices_multivariate.csv')
        self.assertIsInstance(X_train, pandas.core.frame.DataFrame,
                              "Expected data type for return value is `pandas DataFrame`, you are returning %s" % (
                                  type(X_train)))

    def test_load_data_X_test_type(self):  # Return data types
        data, X_train, X_test, y_train, y_test = load_data('data/house_prices_multivariate.csv')
        self.assertIsInstance(X_test, pandas.core.frame.DataFrame,
                              "Expected data type for return value is `pandas DataFrame`, you are returning %s" % (
                                  type(X_test)))

    def test_load_data_y_train_type(self):  # Return data types
        data, X_train, X_test, y_train, y_test = load_data('data/house_prices_multivariate.csv')
        self.assertIsInstance(y_train, pandas.core.series.Series,
                              "Expected data type for return value is `pandas DataFrame`, you are returning %s" % (
                                  type(y_train)))

    def test_load_data_y_test_type(self):  # Return data types
        data, X_train, X_test, y_train, y_test = load_data('data/house_prices_multivariate.csv')
        self.assertIsInstance(y_test, pandas.core.series.Series,
                              "Expected data type for return value is `pandas DataFrame`, you are returning %s" % (
                                  type(y_test)))

    def test_load_data_y_test_values(self):  # Return data values
        data, X_train, X_test, y_train, y_test = load_data('data/house_prices_multivariate.csv')
        self.assertEqual(y_test.shape, (456,),
                         "Return value shape does not match expected value")

    def test_load_data_X_train_values(self):  # Return data values
        data, X_train, X_test, y_train, y_test = load_data('data/house_prices_multivariate.csv')
        self.assertEqual(X_train.shape, (923, 34),
                         "Return value shape does not match expected value")

    def test_load_data_y_train_values(self):  # Return data values
        data, X_train, X_test, y_train, y_test = load_data('data/house_prices_multivariate.csv')
        self.assertEqual(y_train.iloc[4], 113000,
                         "Return value does not match expected value")

    def test_load_data_X_test_values(self):  # Return data values
        data, X_train, X_test, y_train, y_test = load_data('data/house_prices_multivariate.csv')
        self.assertEqual(X_test.iloc[5, 4], 1963,
                         "Return value value does not match expected value")
