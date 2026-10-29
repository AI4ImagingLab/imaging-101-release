"""Unit tests for preprocessing module."""

import os
import unittest
import numpy as np

TASK_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
FIX_DIR = os.path.join(TASK_DIR, 'evaluation', 'fixtures', 'preprocessing')

import sys
sys.path.insert(0, TASK_DIR)


class TestLoadObservation(unittest.TestCase):
    def test_returns_dict_with_expected_keys(self):
        from src.preprocessing import load_observation
        obs = load_observation(os.path.join(TASK_DIR, 'data'))
        required_keys = {'vis_corrupt', 'uv_coords', 'sigma_vis', 'station_ids'}
        self.assertTrue(required_keys.issubset(set(obs.keys())))
        self.assertNotIn('vis_cal', obs)

    def test_shapes(self):
        f = np.load(os.path.join(FIX_DIR, 'load_observation.npz'))
        from src.preprocessing import load_observation
        obs = load_observation(os.path.join(TASK_DIR, 'data'))
        self.assertEqual(obs['vis_corrupt'].shape, (421,))
        np.testing.assert_array_equal(
            np.array(obs['uv_coords'].shape), f['output_uv_coords_shape'])

    def test_values(self):
        from src.preprocessing import load_observation
        obs = load_observation(os.path.join(TASK_DIR, 'data'))
        with np.load(os.path.join(TASK_DIR, 'data', 'raw_data.npz'), allow_pickle=False) as raw:
            np.testing.assert_allclose(obs['vis_corrupt'][:5], raw['vis_corrupt'][0, :5])


class TestLoadMetadata(unittest.TestCase):
    def test_returns_dict(self):
        from src.preprocessing import load_metadata
        meta = load_metadata(os.path.join(TASK_DIR, 'data'))
        expected = np.load(os.path.join(FIX_DIR, 'load_metadata.npz'), allow_pickle=False)
        for key in ('N', 'pixel_size_uas', 'pixel_size_rad', 'total_flux',
                    'freq_ghz', 'n_baselines', 'n_stations'):
            # Acquisition constants must preserve the archived values verbatim.
            self.assertEqual(meta[key], float(expected[f'output_{key}']))


class TestFindTriangles(unittest.TestCase):
    def setUp(self):
        self.f = np.load(os.path.join(FIX_DIR, 'find_triangles.npz'))

    def test_shape(self):
        from src.preprocessing import find_triangles
        tri = find_triangles(self.f['input_station_ids'], int(self.f['input_n_stations']))
        self.assertEqual(tri.ndim, 2)
        self.assertEqual(tri.shape[1], 3)

    def test_count(self):
        from src.preprocessing import find_triangles
        tri = find_triangles(self.f['input_station_ids'], int(self.f['input_n_stations']))
        self.assertEqual(len(tri), len(self.f['output_triangles']))


class TestFindQuadrangles(unittest.TestCase):
    def setUp(self):
        self.f = np.load(os.path.join(FIX_DIR, 'find_quadrangles.npz'))

    def test_shape(self):
        from src.preprocessing import find_quadrangles
        quad = find_quadrangles(self.f['input_station_ids'], int(self.f['input_n_stations']))
        self.assertEqual(quad.ndim, 2)
        self.assertEqual(quad.shape[1], 4)

    def test_count(self):
        from src.preprocessing import find_quadrangles
        quad = find_quadrangles(self.f['input_station_ids'], int(self.f['input_n_stations']))
        self.assertEqual(len(quad), len(self.f['output_quadrangles']))


class TestComputeClosurePhases(unittest.TestCase):
    def setUp(self):
        self.f = np.load(os.path.join(FIX_DIR, 'compute_closure_phases.npz'))

    def test_output_shape(self):
        from src.preprocessing import compute_closure_phases
        cp = compute_closure_phases(
            self.f['input_vis'], self.f['input_station_ids'], self.f['input_triangles'])
        self.assertEqual(cp.shape, self.f['output_cphase'].shape)

    def test_output_values(self):
        from src.preprocessing import compute_closure_phases
        cp = compute_closure_phases(
            self.f['input_vis'], self.f['input_station_ids'], self.f['input_triangles'])
        np.testing.assert_allclose(cp, self.f['output_cphase'], rtol=1e-10)

    def test_output_range(self):
        from src.preprocessing import compute_closure_phases
        cp = compute_closure_phases(
            self.f['input_vis'], self.f['input_station_ids'], self.f['input_triangles'])
        self.assertTrue(np.all(np.abs(cp) <= np.pi))


class TestComputeLogClosureAmplitudes(unittest.TestCase):
    def setUp(self):
        self.f = np.load(os.path.join(FIX_DIR, 'compute_log_closure_amplitudes.npz'))

    def test_output_shape(self):
        from src.preprocessing import compute_log_closure_amplitudes
        lca = compute_log_closure_amplitudes(
            self.f['input_vis'], self.f['input_station_ids'], self.f['input_quadrangles'])
        self.assertEqual(lca.shape, self.f['output_logcamp'].shape)

    def test_output_values(self):
        from src.preprocessing import compute_log_closure_amplitudes
        lca = compute_log_closure_amplitudes(
            self.f['input_vis'], self.f['input_station_ids'], self.f['input_quadrangles'])
        np.testing.assert_allclose(lca, self.f['output_logcamp'], rtol=1e-10)


class TestClosurePhaseSigma(unittest.TestCase):
    def setUp(self):
        self.f = np.load(os.path.join(FIX_DIR, 'closure_phase_sigma.npz'))

    def test_output_values(self):
        from src.preprocessing import closure_phase_sigma
        sigma = closure_phase_sigma(
            self.f['input_sigma_vis'], self.f['input_vis'],
            self.f['input_station_ids'], self.f['input_triangles'])
        np.testing.assert_allclose(sigma, self.f['output_sigma_cp'], rtol=1e-10)

    def test_positive(self):
        from src.preprocessing import closure_phase_sigma
        sigma = closure_phase_sigma(
            self.f['input_sigma_vis'], self.f['input_vis'],
            self.f['input_station_ids'], self.f['input_triangles'])
        self.assertTrue(np.all(sigma > 0))


class TestClosureAmplitudeSigma(unittest.TestCase):
    def setUp(self):
        self.f = np.load(os.path.join(FIX_DIR, 'closure_amplitude_sigma.npz'))

    def test_output_values(self):
        from src.preprocessing import closure_amplitude_sigma
        sigma = closure_amplitude_sigma(
            self.f['input_sigma_vis'], self.f['input_vis'],
            self.f['input_station_ids'], self.f['input_quadrangles'])
        np.testing.assert_allclose(sigma, self.f['output_sigma_lca'], rtol=1e-10)


if __name__ == '__main__':
    unittest.main()
