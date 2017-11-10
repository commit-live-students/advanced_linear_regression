import numpy as np
from unittest import TestCase
from greyatomlib.advanced_linear_regression.q01_load_data.build import load_data
from ..build import Max_important_feature
from inspect import getfullargspec


class TestMax_important_feature(TestCase):
    def test_Max_important_feature(self):
        # Input parameters tests
        args = getfullargspec(Max_important_feature).args
        args_default = getfullargspec(Max_important_feature).defaults
        self.assertEqual(len(args), 3, "Expected argument(s) %d, Given %d" % (3, len(args)))
        self.assertEqual(args_default, ('SalePrice',4), "Expected default values do not match given default values")

        data_set, X_train, X_test, y_train, y_test = load_data('data/house_prices_multivariate.csv')
        arr = Max_important_feature(data_set)
        expected_list = ['OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea']
        # Return value tests
        self.assertListEqual(list(arr), expected_list, "Expected output does not match the given"
                                                       "output")
