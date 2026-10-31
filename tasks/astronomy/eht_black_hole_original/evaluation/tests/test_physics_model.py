"""Unit tests for physics_model module."""

import os
import unittest
import numpy as np

TASK_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
FIX_DIR = os.path.join(TASK_DIR, 'evaluation', 'fixtures', 'physics_model')

import sys
sys.path.insert(0, TASK_DIR)


def _build_model():
    """Build a ClosureForwardModel from the observation data."""
    from src.preprocessing import load_observation, load_metadata, find_triangles, find_quadrangles
    data_dir = os.path.join(TASK_DIR, 'data')
    obs = load_observation(data_dir)
    meta = load_metadata(data_dir)
    tri = find_triangles(obs['station_ids'], meta['n_stations'])
    quad = find_quadrangles(obs['station_ids'], meta['n_stations'])
    from src.physics_model import ClosureForwardModel
    return ClosureForwardModel(
        uv_coords=obs['uv_coords'], N=meta['N'],
        pixel_size_rad=meta['pixel_size_rad'],
        triangles=tri, quadrangles=quad,
        station_ids=obs['station_ids'],
    )


class TestForward(unittest.TestCase):
    def setUp(self):
        self.f = np.load(os.path.join(FIX_DIR, 'forward_unit.npz'))
        self.model = _build_model()

    def test_output_shape(self):
        vis = self.model.forward(self.f['input_image'])
        self.assertEqual(vis.shape[0], self.model.A.shape[0])

    def test_output_dtype(self):
        vis = self.model.forward(self.f['input_image'])
        self.assertTrue(np.iscomplexobj(vis))

    def test_output_values(self):
        vis = self.model.forward(self.f['input_image'])
        np.testing.assert_allclose(vis, self.f['output_vis'], rtol=1e-10)


class TestAdjoint(unittest.TestCase):
    def setUp(self):
        self.f = np.load(os.path.join(FIX_DIR, 'adjoint.npz'))
        self.model = _build_model()

    def test_output_shape(self):
        img = self.model.adjoint(self.f['input_vis'])
        self.assertEqual(img.shape, (self.model.N, self.model.N))

    def test_output_values(self):
        img = self.model.adjoint(self.f['input_vis'])
        np.testing.assert_allclose(img, self.f['output_image'], rtol=1e-9, atol=1e-10)

    def test_adjoint_identity(self):
        """Re[<Ax, y>] = <x, Re[A^H y]> for random real x, complex y."""
        np.random.seed(42)
        x = np.random.randn(self.model.N, self.model.N)
        y = np.random.randn(self.model.A.shape[0]) + 1j * np.random.randn(self.model.A.shape[0])
        lhs = np.vdot(self.model.forward(x), y).real
        rhs = np.vdot(x, self.model.adjoint(y))
        np.testing.assert_allclose(lhs, rhs, rtol=1e-10)


class TestDirtyImage(unittest.TestCase):
    def setUp(self):
        self.f = np.load(os.path.join(FIX_DIR, 'dirty_image.npz'))
        self.model = _build_model()

    def test_output_shape(self):
        dirty = self.model.dirty_image(self.f['input_vis'])
        self.assertEqual(dirty.shape, (self.model.N, self.model.N))

    def test_output_values(self):
        dirty = self.model.dirty_image(self.f['input_vis'])
        np.testing.assert_allclose(dirty, self.f['output_image'], rtol=1e-9, atol=1e-12)


class TestPSF(unittest.TestCase):
    def setUp(self):
        self.f = np.load(os.path.join(FIX_DIR, 'psf.npz'))
        self.model = _build_model()

    def test_output_shape(self):
        psf = self.model.psf()
        self.assertEqual(psf.shape, (self.model.N, self.model.N))

    def test_output_values(self):
        psf = self.model.psf()
        np.testing.assert_allclose(psf, self.f['output_psf'], rtol=1e-9, atol=1e-12)

    def test_peak_at_center(self):
        psf = self.model.psf()
        N = self.model.N
        center = (N // 2, N // 2)
        peak = np.unravel_index(psf.argmax(), psf.shape)
        # Peak should be near center (within 1 pixel)
        self.assertLessEqual(abs(peak[0] - center[0]), 1)
        self.assertLessEqual(abs(peak[1] - center[1]), 1)


if __name__ == '__main__':
    unittest.main()
