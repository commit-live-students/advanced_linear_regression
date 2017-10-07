import sys, os
import numpy as np
sys.path.append(os.path.join(os.path.dirname(os.curdir)))

from unittest import TestCase

from greyatomlib.advanced_linear_regression.q01_load_data.build import load_data
from q02_Max_important_feature.build import Max_important_feature


class TestMax_important_feature(TestCase):
    def test_Max_important_feature(self):
        data_set, X_train, X_test, y_train, y_test = load_data('data/house_prices_multivariate.csv')
        arr = Max_important_feature(data_set, "SalePrice", 4)
        self.assertItemsEqual(arr, np.array(['OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea'], dtype=object))
