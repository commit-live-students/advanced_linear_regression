import os
import sys

import numpy as np

sys.path.append(os.path.join(os.path.dirname(os.curdir)))
from q06_cross_validation.build import cross_validation
from unittest import TestCase
from sklearn.linear_model import Ridge
from greyatomlib.advanced_linear_regression.q01_load_data.build import load_data

data_set, X_train, X_test, y_train, y_test = load_data('data/house_prices_multivariate.csv')


class TestCross_validation(TestCase):
    def test_cross_validation(self):
        np.random.seed(9)

        value = cross_validation(Ridge(alpha=0.1), X_train, y_train)
        self.assertTrue(value, -1778803314.8522613)
