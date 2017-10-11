import numpy as np
from ..build import lasso
from unittest import TestCase
from inspect import getargspec


class TestLasso(TestCase):
    def test_lasso(self):
        np.random.seed(9)

        # Input parameters tests
        args = getargspec(lasso)
        self.assertEqual(len(args[0]), 1, "Expected argument(s) %d, Given %d" % (1, len(args[0])))
        self.assertEqual(args[3], (0.01,), "Expected default values do not match given default values")

        # Return type tests
        rmse1, rmse2 = lasso(0.01)
        self.assertIsInstance(rmse1, float,
                              "Expected data type for return value is `float`, you are returning %s" % (
                                  type(rmse1)))
        self.assertIsInstance(rmse2, float,
                              "Expected data type for return value is `float`, you are returning %s" % (
                                  type(rmse2)))

        # Return value tests
        self.assertAlmostEqual(rmse1, 33769.142311968972, places=3)
        self.assertAlmostEqual(rmse2, 37838.644447277395, places=3)
