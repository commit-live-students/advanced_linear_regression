import os
import sys
import numpy as np

sys.path.append(os.path.join(os.path.dirname(os.curdir)))
from q05_lasso.build import lasso

from unittest import TestCase


class TestLasso(TestCase):
    def test_lasso(self):
        np.random.seed(9)
        rmse1, rmse2 = lasso(0.01)

        self.assertAlmostEqual(rmse1, 33769.142311968972, places=3)
        self.assertAlmostEqual(rmse2, 37838.644447277395, places=3)
