import sys, os

import numpy as np

sys.path.append(os.path.join(os.path.dirname(os.curdir)))

from unittest import TestCase

from q03_polynomial.build import polynomial


class TestPolynomial(TestCase):
    def test_polynomial(self):
        model = polynomial()
        prediction = model.predict(np.array([4, 5, 6, 7]).reshape(1, -1))
        self.assertTrue(prediction, np.array([-50871.05760668]))
       
