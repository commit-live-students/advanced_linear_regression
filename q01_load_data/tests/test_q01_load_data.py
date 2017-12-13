import pandas

from unittest import TestCase
from ..build import load_data
from inspect import getargspec


class TestLoad_data(TestCase):

  def test_load_data(self):   # Input parameters test
    args = getargspec(load_data)
    self.assertEqual(len(args[0]), 3, "Expected argument(s) %d, Given %d" % (3, len(args[0])))

  def test_load_data_default(self):  # Input paramters default
    args = getargspec(load_data) 
    self.assertEqual(args[3], (0.33, 9), "Expected default values do not match given default values")

  def test_load_data_result_types(self):      # Return data types
    
    data, X_train, X_test, y_train, y_test = load_data('data/house_prices_multivariate.csv')
    self.assertIsInstance(X_train, pandas.core.frame.DataFrame,
                              "Expected data type for return value is `pandas DataFrame`, you are returning %s" % (
                                  type(X_train)))

    self.assertIsInstance(X_test, pandas.core.frame.DataFrame,
                              "Expected data type for return value is `pandas DataFrame`, you are returning %s" % (
                                  type(X_test)))

    self.assertIsInstance(y_train, pandas.core.series.Series,
                              "Expected data type for return value is `pandas DataFrame`, you are returning %s" % (
                                  type(y_train)))

    self.assertIsInstance(y_test, pandas.core.series.Series,
                              "Expected data type for return value is `pandas DataFrame`, you are returning %s" % (
                                  type(y_test)))

  def test_load_data_result_values(self):   # Return data values

    data, X_train, X_test, y_train, y_test = load_data('data/house_prices_multivariate.csv')
    self.assertEqual(y_test.shape, (456,),
                         "Return value shape does not match expected value")
    self.assertEqual(X_train.shape, (923, 34),
                         "Return value shape does not match expected value")

    self.assertEqual(y_train.iloc[4], 113000,
                         "Return value does not match expected value")
    self.assertEqual(X_test.iloc[5, 4], 1963,
                         "Return value value does not match expected value")
