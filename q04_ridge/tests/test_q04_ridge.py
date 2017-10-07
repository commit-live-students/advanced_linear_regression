import sys, os
import numpy as np

sys.path.append(os.path.join(os.path.dirname(os.curdir)))
from unittest import TestCase
from q04_ridge.build import ridge

np.random.seed(9)


class TestRidge(TestCase):
    def test_ridge(self):
        rmse1, rmse2 = ridge(0.01)

        self.assertAlmostEqual(rmse1, 33775.6544815, places=3)
        self.assertAlmostEqual(rmse2, 37702.0033295, places=3)
