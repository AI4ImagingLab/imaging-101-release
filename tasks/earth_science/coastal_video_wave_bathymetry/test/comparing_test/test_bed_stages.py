"""Unmodified original B-stage captures, frozen before local extraction."""
from pathlib import Path
import numpy as np
from src.preprocessing import load_data
from src.solvers import fit_bed_point, spatial_uncertainty, fit_bed

FIXTURE = Path(__file__).resolve().parents[1] / 'fixtures/upstream_bed_stage.npz'


def test_original_bfgs_point_cases():
    with np.load(FIXTURE, allow_pickle=False) as f:
        for i in range(8):
            value, info = fit_bed_point(f[f'input_gamma_{i}'], f[f'input_omega_{i}'],
                                      f[f'input_water_{i}'], float(f[f'input_initialbed_{i}'][0]))
            np.testing.assert_allclose(value, f[f'output_bed_{i}'][0], atol=2e-6, rtol=0)
            assert np.isfinite(info['gamma_rmse'])


def test_original_full_spatial_sd_and_nan_erosion():
    xy = load_data()['reconstruction_xy']
    with np.load(FIXTURE, allow_pickle=False) as f:
        raw = f['output_native_zbsB']
        actual = spatial_uncertainty(xy, raw, f['output_native_RBs'])
        expected = f['output_final_std']
        np.testing.assert_array_equal(np.isfinite(actual), np.isfinite(expected))
        np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12, equal_nan=True)
        np.testing.assert_array_equal(np.isfinite(raw) & np.isfinite(actual),
                                      np.isfinite(f['output_final_bed']))


def test_uncertainty_preserves_missing_neighbor():
    xy = np.array([[0., 0.], [1., 0.], [2., 0.], [3., 0.]])
    sd = spatial_uncertainty(xy, np.array([1., 2., np.nan, 4.]), np.full(4, 2.1))
    assert np.isnan(sd).all()


def test_full_deterministic_original_bed_stage():
    with np.load(FIXTURE, allow_pickle=False) as f:
        observations = {
            'xy': np.column_stack([f['output_native_xsKW'], f['output_native_ysKW']]),
            'gamma': f['output_native_gammasKW'], 'mean_gamma': f['output_native_meanGsKW'],
            'omega': f['output_native_wsKW'], 'water_level': f['output_native_ztsKW'],
        }
        prediction, diagnostics = fit_bed(load_data()['reconstruction_xy'], observations)
        expected = f['output_final_bed']
        valid = np.isfinite(expected)
        np.testing.assert_array_equal(prediction['validity'][0], valid)
        np.testing.assert_allclose(prediction['reconstruction'][0, valid], expected[valid], atol=2e-6, rtol=0)
        np.testing.assert_allclose(diagnostics['raw_bed'], f['output_native_zbsB'], atol=2e-6, rtol=0, equal_nan=True)
        np.testing.assert_allclose(diagnostics['radii'], f['output_native_RBs'], atol=1e-10, rtol=1e-12, equal_nan=True)
