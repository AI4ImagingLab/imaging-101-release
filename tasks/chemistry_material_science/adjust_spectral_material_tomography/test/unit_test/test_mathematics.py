from pathlib import Path
import numpy as np
import pytest
from scipy import sparse
from scipy.optimize import nnls, minimize
from src.physics_model import build_system_matrix, geometry_vectors, forward, adjoint, sirt
from src.preprocessing import correct_counts, load_observations, calibrated_dictionary
from src.solvers import (LinearMap, initial_state, project_simplex_columns,
                         project_material_maps, project_dictionary_weights,
                         spectral_objective, spatial_objective,
                         nonnegative_least_squares_small, tikhonov_cg,
                         factor_nonnegative, sequential_comparison)
from src.visualization import numeric_metrics, align_components


def test_axis_ray_lengths_and_support():
    vectors = np.array([[0., -1000., 0., 1000., 1., 0.]])
    matrix = build_system_matrix(vectors, size=9, detector_count=9)
    middle = matrix.toarray()[4].reshape(9, 9, order='F')
    expected = np.zeros((9, 9))
    expected[:, 4] = 1
    np.testing.assert_array_equal(middle, expected)
    assert np.all(matrix.data >= 0)
    assert np.isclose(matrix[4].sum(), 9)


def test_exact_discrete_adjoint():
    rng = np.random.default_rng(19)
    matrix = sparse.random(31, 17, density=.4, random_state=rng)
    x, y = rng.normal(size=(17, 3)), rng.normal(size=(31, 3))
    left, right = np.sum(forward(matrix, x) * y), np.sum(x * adjoint(matrix, y))
    assert abs(left - right) / max(abs(left), abs(right)) < 1e-13


def test_geometry_units_orientation_and_inclusive_endpoint():
    vectors = geometry_vectors(np.array([0., np.pi / 2, 2 * np.pi]))
    assert vectors[0, 1] < 0
    assert vectors[1, 0] > 0
    assert abs(vectors[0, 0]) == 0
    np.testing.assert_allclose(vectors[0], vectors[2], atol=1e-10)
    np.testing.assert_allclose(np.linalg.norm(vectors[:, 4:], axis=1), 883.081116 / 57.999001)


def test_log_counts_unsigned_subtraction_and_order():
    counts = np.array([[[2, 3]], [[4, 5]]], dtype=np.uint16)
    flat, dark = np.array([[9, 10]], np.uint16), np.array([[1, 1]], np.uint16)
    result = correct_counts(counts, dark, flat)
    expected = np.array([[np.log(8 / 3)], [np.log(8)], [np.log(9 / 4)], [np.log(9 / 2)]])
    np.testing.assert_allclose(result, expected)
    with pytest.raises(ValueError):
        correct_counts(np.ones_like(counts), dark, flat)


def test_public_observation_contract():
    observation = load_observations()
    assert observation['counts'].dtype == np.uint16
    assert observation['counts'].shape == (600, 128, 257)
    assert observation['angles'][0] == 0
    assert observation['angles'][-1] == 2 * np.pi
    assert observation['dictionary'].shape == (3, 128)
    assert np.all(np.isfinite(correct_counts(observation['counts'], observation['dark'], observation['flat'])))


def test_readme_acquisition_constants_match_executable_geometry():
    from src import physics_model as physics
    readme = (Path(__file__).resolve().parents[2] / 'README.md').read_text(encoding='utf-8')
    rows = {cells[0].strip(): cells[1].strip()
            for line in readme.splitlines() if line.startswith('|')
            for cells in [line.strip('|').split('|')] if len(cells) == 4}
    for key, expected in [('Source-object distance', physics.SOURCE_OBJECT_MM),
                          ('Source-detector distance', physics.SOURCE_DETECTOR_MM),
                          ('Detector pitch', physics.DETECTOR_PITCH_MM),
                          ('Detector positions', physics.DETECTOR_COUNT),
                          ('Acquired projections', physics.ANGLE_COUNT)]:
        assert float(rows[key]) == expected
    observation = load_observations()
    np.testing.assert_allclose(observation['energies'], 20.90 + 1.10 * np.arange(128), atol=3e-14)
    np.testing.assert_array_equal(observation['angles'], np.linspace(0, 2 * np.pi, 600))


def test_quartz_roi_is_actively_derived_with_one_based_indices():
    channels = np.zeros((65025, 128), dtype=np.float32)
    image = np.zeros((255, 255), np.float32)
    image[129:140, 129:140] = 2
    channels[:, 0] = image.ravel(order='F')
    dictionary, quartz = calibrated_dictionary(np.ones((3, 128)), channels)
    assert quartz[0] == 2
    assert np.sum(quartz[1:]) == 0
    assert dictionary.shape == (4, 65)
    np.testing.assert_allclose(dictionary[:3], .015)


def test_simplex_projection_independent_quadratic_program():
    values = np.array([[2., -.3, .1], [-.5, .9, .2], [.2, .8, .1]])
    answer = project_simplex_columns(values)
    for column in range(3):
        x = values[:, column]
        check = minimize(lambda z: .5 * np.sum((z - x) ** 2), np.ones(3) / 3,
                         jac=lambda z: z - x, bounds=[(0, None)] * 3,
                         constraints=[{'type': 'ineq', 'fun': lambda z: 1 - z.sum()}],
                         method='SLSQP', options={'ftol': 1e-13})
        assert check.success
        np.testing.assert_allclose(answer[:, column], check.x, atol=1e-7)


def test_repaired_dictionary_guard_and_feasibility():
    # The original sum<1 skip returned these negative entries unchanged.
    result = project_dictionary_weights(np.full((2, 3), -.5))
    np.testing.assert_allclose(result, 1 / 3, atol=1e-15)
    np.testing.assert_allclose(result.sum(axis=1), 1, atol=1e-6)
    assert np.all(result >= 0)
    assert np.max(result.sum(axis=0)) <= 1 + 1e-6


def test_feasibility_map_does_not_claim_orthogonal_projection():
    values = np.array([[2., -1., .2], [1.8, .6, -1.]])
    result = project_dictionary_weights(values)
    nearest = np.array([[.65, 0., .35], [.35, .65, 0.]])
    assert np.max(abs(result - nearest)) > .39
    np.testing.assert_allclose(result.sum(axis=1), 1, atol=1e-6)
    assert np.max(result.sum(axis=0)) <= 1 + 1e-6
    assert np.min(result) >= 0


def test_gradient_directional_derivatives():
    rng = np.random.default_rng(41)
    matrix = sparse.csr_matrix(rng.normal(size=(21, 13)))
    op = LinearMap(matrix)
    maps, weights = rng.normal(size=(13, 4)), rng.normal(size=(4, 7))
    dictionary, y = rng.normal(size=(7, 9)), rng.normal(size=(21, 9))
    norm = np.linalg.norm(y)
    dr, da = rng.normal(size=weights.shape), rng.normal(size=maps.shape)
    f, gr = spectral_objective(weights, y, matrix @ maps, dictionary, norm)
    f1 = spectral_objective(weights + 1e-6 * dr, y, matrix @ maps, dictionary, norm)[0]
    f0 = spectral_objective(weights - 1e-6 * dr, y, matrix @ maps, dictionary, norm)[0]
    np.testing.assert_allclose((f1 - f0) / 2e-6, np.sum(gr * dr), rtol=2e-8)
    f, ga = spatial_objective(maps, weights @ dictionary, y, op, norm)
    f1 = spatial_objective(maps + 1e-6 * da, weights @ dictionary, y, op, norm)[0]
    f0 = spatial_objective(maps - 1e-6 * da, weights @ dictionary, y, op, norm)[0]
    np.testing.assert_allclose((f1 - f0) / 2e-6, np.sum(ga * da), rtol=2e-8)


def test_sirt_update_and_zero_rows():
    matrix = sparse.csr_matrix(np.array([[1., 0.], [0., 2.], [0., 0.]], dtype=np.float32))
    y = np.array([[2.], [6.], [0.]], np.float32)
    np.testing.assert_allclose(sirt(matrix, y, iterations=1), [[2.], [3.]])


def test_general_operator_bad_backend():
    with pytest.raises(ValueError):
        LinearMap(sparse.eye(3), backend='not-a-device')


def test_regularized_cg_independent_direct_solution():
    matrix = sparse.csr_matrix([[1., 2.], [3., 1.], [2., 4.]])
    y = np.array([[1., 3.], [3., 2.], [5., 1.]])
    result = tikhonov_cg(y, matrix)
    exact = np.linalg.solve((matrix.T @ matrix).toarray() + .001 * np.eye(2), matrix.T @ y)
    np.testing.assert_allclose(result, exact, rtol=1e-11, atol=1e-11)


def test_small_nnls_against_independent_scipy():
    rng = np.random.default_rng(5)
    matrix = rng.normal(size=(17, 4))
    data = rng.normal(size=(17, 12))
    result = nonnegative_least_squares_small(matrix.T @ matrix, matrix.T @ data)
    expected = np.stack([nnls(matrix, column)[0] for column in data.T], axis=1)
    np.testing.assert_allclose(result, expected, rtol=1e-10, atol=1e-12)
    with pytest.raises(ValueError):
        nonnegative_least_squares_small(np.eye(5), np.ones((5, 2)))


def test_sequential_factorization_and_order_contract():
    rng = np.random.default_rng(8)
    data = rng.random((16, 4)) @ rng.random((4, 9))
    maps, spectra = factor_nonnegative(data, replicates=2, multiplicative_iterations=10, als_iterations=5)
    assert np.min(maps) >= 0 and np.min(spectra) >= 0
    assert np.linalg.norm(data - maps @ spectra) < .12 * np.linalg.norm(data)
    with pytest.raises(ValueError):
        sequential_comparison(data, sparse.eye(16), order='invalid')


def test_metrics_known_case_and_permutation():
    result = numeric_metrics([1., 2., 3.], [3., 2., 1.])
    assert result['ncc'] == pytest.approx(10 / 14)
    assert result['nrmse'] == pytest.approx(np.sqrt(8 / 3) / 2)
    spectra = np.eye(4, 7)
    np.testing.assert_array_equal(align_components(spectra[[2, 0, 3, 1]], spectra), [1, 3, 0, 2])
    with pytest.raises(ValueError):
        numeric_metrics(np.ones(3), np.ones(3))
