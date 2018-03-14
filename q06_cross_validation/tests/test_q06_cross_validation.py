import numpy as np
from ..build import cross_validation
from unittest import TestCase
from sklearn.linear_model import Ridge
from greyatomlib.advanced_linear_regression.q01_load_data.build import load_data
from inspect import getfullargspec

#  data loading
data_set, X_train, X_test, y_train, y_test = load_data('data/house_prices_multivariate.csv')
np.random.seed(9)

class TestCross_validation(TestCase):
    def test_cross_validation(self):   # Test arguments
        args = getfullargspec(cross_validation)
        self.assertEqual(len(args[0]), 3, "Expected argument(s) %d, Given %d" % (3, len(args)))

    def test_cross_validation_default(self):
        args = getfullargspec(cross_validation)
        self.assertEqual(args[3], None, "Expected default values do not match given default values")

    def test_cross_validation_result_type(self):# Return type tests
        value = cross_validation(Ridge(alpha=0.1), X_train, y_train)
        finalscore = value.mean()
        self.assertIsInstance(finalscore, float,
                              "Expected data type for return value is `float`, you are returning %s" % (
                                  type(finalscore)))


    def test_cross_validation_result_values(self):#  Return type value
        value = cross_validation(Ridge(alpha=0.1), X_train, y_train)
        self.assertAlmostEqual(value, -1764180038.71, 2, "Expected value does not match given value")
