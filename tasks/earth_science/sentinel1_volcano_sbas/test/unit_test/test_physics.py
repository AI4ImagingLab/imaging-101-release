import numpy as np
import pytest

from src.physics_model import network_matrices, temporal_model, closure_integers
from src.solvers import estimate_pixel, lasso_ambiguities, deramp_timeseries


def test_triangle_and_integrated_velocity_are_compatible():
    dates = np.array([20200101, 20200113, 20200125])
    pairs = np.array([[0, 1], [0, 2], [1, 2]])
    a, b, c, delta = network_matrices(dates, pairs)
    np.testing.assert_array_equal(c @ a, 0)
    velocity = np.array([[2.], [-1.]])
    disp = np.vstack(([[0.]], np.cumsum(velocity * delta, axis=0)))
    np.testing.assert_allclose(a @ disp[1:], b @ velocity, atol=1e-8)
    recovered, q, n = estimate_pixel(a, b, b @ velocity, delta, np.ones((3, 1)))
    np.testing.assert_allclose(recovered, disp, atol=2e-8)
    assert q == pytest.approx(1.) and n == 3


def test_closure_detects_integer_cycle_and_ignores_small_noise():
    c = np.array([[1., -1., 1.]])
    phi = np.array([[0.1], [.3], [.2 + 2 * np.pi]])
    np.testing.assert_array_equal(closure_integers(c, phi), [[1.]])


def test_lasso_satisfies_kkt_on_unique_case():
    a = np.eye(4)
    y = np.array([[2.], [-1.], [0.], [.002]])
    solution, diagnostics = lasso_ambiguities(a, y, alpha=.01)
    np.testing.assert_allclose(solution, np.sign(y) * np.maximum(np.abs(y) - .005, 0), atol=2e-6)
    assert diagnostics['kkt_residual'] < 1e-5


def test_deramp_removes_reference_anchored_plane():
    yy, xx = np.mgrid[:12, :13]
    data = np.stack((2 * (yy - 3) + 4 * (xx - 5), -3 * (yy - 3) + (xx - 5))).astype(np.float32)
    out, ramp = deramp_timeseries(data, np.ones((12, 13), bool), reference=(3, 5))
    np.testing.assert_allclose(out, 0, atol=2e-5)
    np.testing.assert_array_equal(ramp[:, 3, 5], 0)


def test_eruption_model_has_acceleration_factor_and_strict_step():
    dates = np.array([20170909, 20170910, 20170911, 20180613, 20180614])
    g = temporal_model(dates)
    assert g.shape == (5, 5)
    np.testing.assert_allclose(g[:, 2], .5 * g[:, 1] ** 2, atol=1e-6)
    np.testing.assert_array_equal(g[:, 3], [0, 0, 1, 1, 1])
    np.testing.assert_array_equal(g[:, 4], [0, 0, 0, 0, 1])


def test_inadequate_network_returns_explicit_zero_quality():
    a, b, _, delta = network_matrices(np.array([20200101, 20200113, 20200125]), np.array([[0, 1], [0, 2], [1, 2]]))
    x, q, n = estimate_pixel(a, b, np.full((3, 1), np.nan), delta, np.ones((3, 1)))
    np.testing.assert_array_equal(x, 0)
    assert q == 0 and n == 0
