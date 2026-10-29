"""Unit tests for visualization module."""

import os
import unittest
import numpy as np

TASK_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
FIX_DIR = os.path.join(TASK_DIR, 'evaluation', 'fixtures', 'visualization')

import sys
sys.path.insert(0, TASK_DIR)


class TestComputeMetrics(unittest.TestCase):
    def setUp(self):
        f = np.load(os.path.join(FIX_DIR, 'compute_metrics.npz'))
        self.estimate = f['input_estimate']
        self.reference = f['input_reference']
        self.expected_nrmse = float(f['output_nrmse'])
        self.expected_ncc = float(f['output_ncc'])

    def test_nrmse(self):
        from src.visualization import compute_metrics
        m = compute_metrics(self.estimate, self.reference)
        np.testing.assert_allclose(m['nrmse'], self.expected_nrmse, rtol=1e-3)

    def test_ncc(self):
        from src.visualization import compute_metrics
        m = compute_metrics(self.estimate, self.reference)
        np.testing.assert_allclose(m['ncc'], self.expected_ncc, rtol=1e-3)

    def test_perfect_match(self):
        from src.visualization import compute_metrics
        m = compute_metrics(self.reference.copy(), self.reference)
        self.assertAlmostEqual(m['ncc'], 1.0, places=3)
        self.assertAlmostEqual(m['nrmse'], 0.0, places=3)

    def test_returns_dict(self):
        from src.visualization import compute_metrics
        m = compute_metrics(self.estimate, self.reference)
        self.assertIn('nrmse', m)
        self.assertIn('ncc', m)
        self.assertIn('dynamic_range', m)


if __name__ == '__main__':
    unittest.main()
