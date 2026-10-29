"""Unit tests for solvers module."""

import os
import unittest
import numpy as np

TASK_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
FIX_DIR = os.path.join(TASK_DIR, 'evaluation', 'fixtures', 'solvers')

import sys
sys.path.insert(0, TASK_DIR)


class TestGullSkillingRegularizer(unittest.TestCase):
    def setUp(self):
        f = np.load(os.path.join(FIX_DIR, 'regularizers.npz'))
        self.test_img = f['input_image']
        self.prior = f['prior_image']
        self.expected_val = float(f['gs_val'])
        self.expected_grad = f['gs_grad']

    def test_value(self):
        from src.solvers import GullSkillingRegularizer
        reg = GullSkillingRegularizer(prior=self.prior)
        val, _ = reg.value_and_grad(self.test_img)
        np.testing.assert_allclose(val, self.expected_val, rtol=1e-10)

    def test_grad(self):
        from src.solvers import GullSkillingRegularizer
        reg = GullSkillingRegularizer(prior=self.prior)
        _, grad = reg.value_and_grad(self.test_img)
        np.testing.assert_allclose(grad.ravel(), self.expected_grad, rtol=1e-10)


class TestSimpleEntropyRegularizer(unittest.TestCase):
    def setUp(self):
        f = np.load(os.path.join(FIX_DIR, 'regularizers.npz'))
        self.test_img = f['input_image']
        self.prior = f['prior_image']
        self.expected_val = float(f['simple_val'])
        self.expected_grad = f['simple_grad']

    def test_value(self):
        from src.solvers import SimpleEntropyRegularizer
        reg = SimpleEntropyRegularizer(prior=self.prior)
        val, _ = reg.value_and_grad(self.test_img)
        np.testing.assert_allclose(val, self.expected_val, rtol=1e-10)

    def test_grad(self):
        from src.solvers import SimpleEntropyRegularizer
        reg = SimpleEntropyRegularizer(prior=self.prior)
        _, grad = reg.value_and_grad(self.test_img)
        np.testing.assert_allclose(grad.ravel(), self.expected_grad, rtol=1e-10)


class TestTVRegularizer(unittest.TestCase):
    def setUp(self):
        f = np.load(os.path.join(os.path.join(FIX_DIR, 'tv_regularizer.npz')))
        self.test_img = f['input_image']
        self.expected_val = float(f['output_val'])
        self.expected_grad = f['output_grad']

    def test_value(self):
        from src.solvers import TVRegularizer
        reg = TVRegularizer(epsilon=1e-6)
        val, _ = reg.value_and_grad(self.test_img)
        np.testing.assert_allclose(val, self.expected_val, rtol=1e-10)

    def test_grad(self):
        from src.solvers import TVRegularizer
        reg = TVRegularizer(epsilon=1e-6)
        _, grad = reg.value_and_grad(self.test_img)
        np.testing.assert_allclose(grad, self.expected_grad, rtol=1e-10)

    def test_zero_image_gives_zero_val(self):
        from src.solvers import TVRegularizer
        reg = TVRegularizer(epsilon=1e-6)
        val, _ = reg.value_and_grad(np.zeros((64, 64)))
        self.assertAlmostEqual(val, 0.0, places=8)


if __name__ == '__main__':
    unittest.main()
