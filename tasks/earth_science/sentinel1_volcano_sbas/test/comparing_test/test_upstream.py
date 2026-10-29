from pathlib import Path
import numpy as np

from src.physics_model import network_matrices, temporal_model
from src.solvers import estimate_pixel, lasso_ambiguities, repair_closure, unweighted_common_baseline, deramp_timeseries, correct_dem
from src.preprocessing import load_data, reference_atmosphere
from src.physics_model import dem_design

ROOT = Path(__file__).resolve().parents[2]


def test_frozen_upstream_network_and_time_basis():
    data = load_data(ROOT / 'data/raw_data.npz')
    with np.load(ROOT / 'test/fixtures/upstream_stages.npz', allow_pickle=False) as f:
        a, b, c, delta = network_matrices(data['dates'], data['pairs'])
        for local, key in [(a, 'A'), (b, 'B'), (c, 'C'), (delta, 'tbase_diff')]:
            np.testing.assert_array_equal(local, f[key])
        np.testing.assert_array_equal(temporal_model(data['dates']), f['G'][:, 1:])


def test_frozen_upstream_weighted_svd_on_real_pixels():
    with np.load(ROOT / 'test/fixtures/upstream_stages.npz', allow_pickle=False) as f:
        for j in range(len(f['coords'])):
            phi = f['unwrapPhase_phaseClosure'][:, j:j+1].copy()
            phi[f['connectComponent'][:, j:j+1] == 0] = np.nan
            weight = np.sqrt(np.maximum(f['coherence'][:, j:j+1], .05))
            x, q, _ = estimate_pixel(f['A'], f['B'], phi, f['tbase_diff'], weight)
            np.testing.assert_allclose(x[:, 0] * np.float32(-.05546576 / (4*np.pi)), f['network'][:, j], atol=2e-7, rtol=2e-6)
            np.testing.assert_allclose(q, f['quality'][j], atol=2e-6)


def test_frozen_upstream_lasso_integer_solution_and_objective():
    with np.load(ROOT/'test/fixtures/upstream_lasso.npz', allow_pickle=False) as f:
        result, diagnostic = lasso_ambiguities(f['input_matrix'], f['input_cycles'])
        np.testing.assert_array_equal(np.median(np.round(result), axis=1), f['output_median'])
        # A rank-deficient closure graph permits different optimal edge vectors.
        # The scientific correction is the region median, checked above.
        from scipy.linalg import null_space
        changed = np.flatnonzero(np.any(np.round(result) != f['output_integer'], axis=0))
        for j in changed:
            support = np.flatnonzero(np.abs(result[:, j]) > 1e-5)
            directions = null_space(f['input_matrix'][:, support])
            assert directions.shape[1] > 0
            direction = directions[:, 0]
            step = .25 * np.min(np.abs(result[support, j][np.abs(direction) > 1e-12] / direction[np.abs(direction) > 1e-12]))
            alternate = result[:, j].copy()
            alternate[support] += step * direction
            np.testing.assert_allclose(f['input_matrix'] @ alternate, f['input_matrix'] @ result[:, j], atol=1e-12)
            np.testing.assert_allclose(np.abs(alternate).sum(), np.abs(result[:, j]).sum(), atol=1e-12)
        objective = lambda x: np.sum((f['input_matrix']@x-f['input_cycles'])**2, axis=0)+.01*np.abs(x).sum(axis=0)
        assert np.max(objective(result)-objective(f['output_solution'])) < 1e-6
        assert diagnostic['kkt_residual'] < 1e-7


def test_full_spatial_closure_and_uniform_sbas_upstream_parity():
    data = load_data(ROOT/'data/raw_data.npz')
    a, b, c, delta = network_matrices(data['dates'], data['pairs'])
    corrected, regions = repair_closure(data['phase'], data['components'], c)
    with np.load(ROOT/'test/fixtures/upstream_regions.npz', allow_pickle=False) as f:
        for i, region in enumerate(regions):
            np.testing.assert_array_equal(region['ambiguity'], f[f'ambiguity_{i}'])
    with np.load(ROOT/'test/fixtures/upstream_stages.npz', allow_pickle=False) as f:
        for j, (y, x) in enumerate(f['coords']):
            np.testing.assert_array_equal(corrected[:, y, x], f['unwrapPhase_phaseClosure'][:, j])
    uniform, quality = unweighted_common_baseline(corrected, data['analysis_mask'], b, delta)
    with np.load(ROOT/'notebooks/reference_outputs/upstream_uniform_sbas.npz', allow_pickle=False) as f:
        np.testing.assert_allclose(uniform[f['frames']], f['network_maps'], atol=2e-7, rtol=2e-6)
        np.testing.assert_allclose(quality, f['temporal_coherence'], atol=3e-6)


def test_upstream_correction_stage_snapshots_on_full_bound_history():
    with np.load(ROOT/'data/raw_data.npz', allow_pickle=False) as raw:
        atmosphere = reference_atmosphere(raw['atmospheric_delay'][0])
        design = dem_design(raw['baseline'][0], raw['incidence'][0], raw['dates'][0])
    with np.load(ROOT/'output/reconstruction.npz', allow_pickle=False) as fresh:
        network, quality = fresh['network_timeseries'][0], fresh['temporal_coherence'][0]
    # Observation availability is known directly, independent of any reference reconstruction.
    with np.load(ROOT/'data/raw_data.npz', allow_pickle=False) as raw:
        fit_mask = raw['analysis_mask'][0] & (quality >= .7)
    tropo = network - atmosphere
    deramped, ramp = deramp_timeseries(tropo, fit_mask)
    final, _, _, residual = correct_dem(deramped, design)
    with np.load(ROOT/'notebooks/reference_outputs/upstream_measurements.npz', allow_pickle=False) as original:
        frames = original['frames']
        for key, value in [('uncorrected', network), ('troposphere_corrected', tropo), ('deramped', deramped), ('reconstruction', final), ('residual', residual), ('ramp', ramp)]:
            np.testing.assert_allclose(value[frames], original[key], atol=2e-7, rtol=2e-6, err_msg=key)
