import numpy as np
from unittest import TestCase
from ..build import polynomial
from inspect import getargspec


class TestPolynomial(TestCase):
    def test_polynomial(self):  # Input parameters tests
        args = getargspec(polynomial)
        self.assertEqual(len(args[0]), 2, "Expected argument(s) %d, Given %d" % (2, len(args)))

    def test_polynomial_default(self):  # Input parameter defaults
    	args = getargspec(polynomial)
        self.assertEqual(args[3], (5, 9), "Expected default values do not match given default values")

    def test_polynomial_result_values(self):  # Return value tests
        model = polynomial()
        prediction = model.predict(np.array([4, 5, 6, 7]).reshape(1, -1))
        self.assertEqual(np.round_(prediction,2), np.array([np.round_(32740.9,2)]))

