import numpy as np
from ..build import cross_validation
from unittest import TestCase
from sklearn.linear_model import Ridge
from greyatomlib.advanced_linear_regression.q01_load_data.build import load_data
from inspect import getfullargspec

#  data loading
data_set, X_train, X_test, y_train, y_test = load_data('data/house_prices_multivariate.csv')


class TestCross_validation(TestCase):
    def test_cross_validation(self):
        np.random.seed(9)
        args = getfullargspec(cross_validation).args
        args_default = getfullargspec(cross_validation).defaults
        self.assertEqual(len(args), 3, "Expected argument(s) %d, Given %d" % (3, len(args)))
        self.assertEqual(args_default, None, "Expected default values do not match given default values")

        # Return type tests
        value = cross_validation(Ridge(alpha=0.1), X_train, y_train)
        finalscore = value.mean()
        self.assertIsInstance(finalscore, float,
                              "Expected data type for return value is `float`, you are returning %s" % (
                                  type(finalscore)))

        # Return value tests
        self.assertAlmostEqual(value, -1764180038.71, 2, "Expected value does not match given value")
