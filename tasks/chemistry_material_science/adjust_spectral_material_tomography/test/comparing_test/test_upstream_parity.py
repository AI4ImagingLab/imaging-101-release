from pathlib import Path
import numpy as np
import pytest
from scipy import sparse
from src.physics_model import build_system_matrix, geometry_vectors, sirt
from src.preprocessing import load_observations, correct_counts
from src.solvers import (initial_state, LinearMap, spectral_objective,
                         spatial_objective, adjust)

FIXTURES = Path(__file__).resolve().parents[1] / 'fixtures'


def test_frozen_upstream_line_weights_and_products():
    with np.load(FIXTURES / 'upstream_line_operator.npz', allow_pickle=False) as reference:
        matrix = build_system_matrix(geometry_vectors(reference['param_angles']))
        expected = sparse.csr_matrix((reference['param_weights'], reference['param_indices'],
                                      reference['param_indptr']), shape=matrix.shape)
        assert (matrix - expected).nnz == 0
        # Preserve the independently captured traversal, not just equal weights.
        np.testing.assert_array_equal(matrix.indptr, reference['param_indptr'])
        np.testing.assert_array_equal(matrix.indices, reference['param_indices'])
        np.testing.assert_array_equal(matrix.data, reference['param_weights'])
        np.testing.assert_allclose(matrix @ reference['input_maps'], reference['output_forward'], rtol=2e-13, atol=2e-12)
        np.testing.assert_allclose(matrix.T @ reference['input_sinograms'], reference['output_adjoint'], rtol=2e-13, atol=2e-12)


def test_independent_original_objectives_gradients_and_two_outer_steps():
    with np.load(FIXTURES / 'upstream_stages.npz', allow_pickle=False) as r:
        matrix = sparse.csr_matrix(r['param_system'])
        data, dictionary = r['input_measurements'], r['param_dictionary']
        maps, weights = initial_state(matrix.shape[1], 4, len(dictionary))
        np.testing.assert_array_equal(maps, r['output_initial_maps'])
        np.testing.assert_array_equal(weights, r['output_initial_weights'])
        f, grad = spectral_objective(weights, data, matrix @ maps, dictionary, np.linalg.norm(data))
        np.testing.assert_allclose(f, r['output_spectral_objective'].item(), rtol=2e-13)
        np.testing.assert_allclose(grad.ravel(order='F'), r['output_spectral_gradient'].ravel(), rtol=2e-12, atol=2e-11)
        _, grad = spatial_objective(maps, weights @ dictionary, data, LinearMap(matrix), np.linalg.norm(data))
        np.testing.assert_allclose(grad.ravel(order='F'), r['output_spatial_gradient'].ravel(), rtol=2e-12, atol=2e-11)
    with np.load(FIXTURES / 'upstream_corrected_stages.npz', allow_pickle=False) as corrected:
        result = adjust(data, matrix, dictionary, iterations=2, verbose=False)
        np.testing.assert_allclose(result['maps'], corrected['maps'], rtol=2e-11, atol=2e-12)
        np.testing.assert_allclose(result['spectra'], corrected['spectra'], rtol=2e-11, atol=2e-12)
        np.testing.assert_allclose(result['weights'], corrected['weights'], rtol=2e-11, atol=2e-12)


@pytest.mark.parametrize('backend', ['cpu', 'cuda_ordered'])
def test_original_cpu_sirt_on_full_measured_geometry(backend):
    if backend == 'cuda_ordered':
        cp = pytest.importorskip('cupy')
        if cp.cuda.runtime.getDeviceCount() == 0:
            pytest.skip('ordered CUDA calibration requires a GPU')
    observations = load_observations()
    matrix = build_system_matrix(geometry_vectors(observations['angles']))
    data = correct_counts(observations['counts'], observations['dark'], observations['flat'])[:, :2]
    reconstructed = sirt(matrix, data, iterations=100, backend=backend)
    with np.load(FIXTURES / 'upstream_sirt.npz', allow_pickle=False) as reference:
        expected = np.stack([image.ravel(order='F') for image in reference['output_first_two_channels']], axis=1)
        relative = np.linalg.norm(reconstructed - expected) / np.linalg.norm(expected)
        assert relative < 2e-6
        assert np.max(abs(reconstructed - expected)) < 5e-7
        np.testing.assert_array_equal(reconstructed, expected)


def test_four_material_original_stage_fixture():
    """Independent source arithmetic on real-data/operator subarrays, not a full run."""
    with np.load(FIXTURES / 'upstream_four_stages.npz', allow_pickle=False) as r:
        matrix = sparse.csr_matrix(r['W'])
        data, dictionary = r['Y'], r['T']
        maps, weights = initial_state(matrix.shape[1], 4, 4)
        np.testing.assert_array_equal(maps, r['A'])
        np.testing.assert_array_equal(weights, r['R'])
        f, gradient = spectral_objective(weights, data, matrix @ maps, dictionary, np.linalg.norm(data))
        np.testing.assert_allclose(f, r['fR'].item(), rtol=2e-10, atol=2e-11)
        np.testing.assert_allclose(gradient.ravel(order='F'), r['gR'].ravel(), rtol=2e-10, atol=2e-11)
        f, gradient = spatial_objective(maps, weights @ dictionary, data, LinearMap(matrix), np.linalg.norm(data))
        np.testing.assert_allclose(f, r['fA'].item(), rtol=2e-10, atol=2e-11)
        np.testing.assert_allclose(gradient.ravel(order='F'), r['gA'].ravel(), rtol=2e-10, atol=2e-11)
        result = adjust(data, matrix, dictionary, iterations=2, verbose=False)
        for actual, expected in [('maps', 'Ac'), ('spectra', 'Fc'), ('weights', 'Rc')]:
            np.testing.assert_allclose(result[actual], r[expected], rtol=2e-10, atol=2e-11)
