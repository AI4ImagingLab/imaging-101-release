import numpy as np
import pytest
from src.physics_model import (wrapped_laplacian, interior_mask, forward_difference,
    backward_difference, laplacian, wave_operator, symmetric_gradient,
    symmetric_divergence, project_balls)
from src.preprocessing import crop_slices, SPACING, PHASE_PER_PPM
from src.solvers import step_sizes, primal_dual


def test_quadratic_laplacian_and_harmonic_background():
    x, y, z = np.meshgrid(np.arange(9)*.2, np.arange(10)*.3, np.arange(11)*.4, indexing='ij')
    spacing = (.2, .3, .4)
    phase = .03*(x*x+y*y+z*z)
    np.testing.assert_allclose(wrapped_laplacian(phase, spacing)[1:-1, 1:-1, 1:-1], .18, atol=1e-12)
    harmonic = .02*(x*x-y*y)+.05*x+.1*y
    np.testing.assert_allclose(wrapped_laplacian(phase+harmonic, spacing)[1:-1, 1:-1, 1:-1], .18, atol=1e-12)


def test_integer_phase_wrap_invariance():
    rng = np.random.default_rng(15)
    phase = rng.normal(size=(11, 12, 13))*.1
    jumps = rng.integers(-3, 4, size=phase.shape)*2*np.pi
    np.testing.assert_allclose(wrapped_laplacian(phase, SPACING), wrapped_laplacian(phase+jumps, SPACING), atol=3e-14)


@pytest.mark.parametrize('axis', range(3))
def test_first_difference_adjoint(axis):
    rng = np.random.default_rng(20+axis)
    u, v = rng.normal(size=(2, 9, 10, 11))
    lhs = np.sum(forward_difference(u, axis, SPACING)*v)
    rhs = -np.sum(u*backward_difference(v, axis, SPACING))
    assert abs(lhs-rhs) < 1e-11


def test_anisotropic_symmetric_frobenius_adjoint():
    rng = np.random.default_rng(173)
    w, q = rng.normal(size=(3, 11, 12, 13)), rng.normal(size=(6, 11, 12, 13))
    weights = np.array([1, 2, 2, 1, 2, 1])[:, None, None, None]
    lhs = np.sum(symmetric_gradient(w, SPACING)*q*weights)
    rhs = -np.sum(w*symmetric_divergence(q, SPACING))
    assert abs(lhs-rhs)/max(abs(lhs), abs(rhs)) < 1e-12


def test_pde_sign_on_independent_fourier_mode():
    n = 24
    k = 2*np.pi*2/n
    x = np.arange(n)[:, None, None]
    chi = np.broadcast_to(np.cos(k*x), (n, n, n)).copy()
    # For a transverse k-vector, D=+1/3 and W=-Laplacian/3.
    np.testing.assert_allclose(laplacian(chi/3, (1, 1, 1)), -wave_operator(chi, (1, 1, 1)), atol=1e-14)


def test_symmetric_regularizer_directional_derivative():
    rng = np.random.default_rng(41)
    w, v = rng.normal(size=(2, 3, 9, 10, 11))
    weights = np.array([1, 2, 2, 1, 2, 1])[:, None, None, None]
    energy = lambda u: .5*np.sum(symmetric_gradient(u, SPACING)**2*weights)
    grad = -symmetric_divergence(symmetric_gradient(w, SPACING), SPACING)
    eps = 1e-5
    fd = (energy(w+eps*v)-energy(w-eps*v))/(2*eps)
    assert abs(fd-np.sum(grad*v))/max(1, abs(fd)) < 1e-8


@pytest.mark.parametrize('tensor', [False, True])
def test_dual_projection_feasibility_and_identity(tensor):
    rng = np.random.default_rng(3)
    v = rng.normal(size=((6 if tensor else 3), 5, 6, 7))
    projected = project_balls(v, .3, tensor)
    weights = np.array([1, 2, 2, 1, 2, 1]) if tensor else np.ones(3)
    norm = np.sqrt(np.sum(projected**2*weights[:, None, None, None], axis=0))
    assert norm.max() <= .3+1e-15
    np.testing.assert_allclose(project_balls(projected, .3, tensor), projected, atol=1e-15)


def test_crop_and_upstream_center_hole_convention():
    mask = np.zeros((12, 13, 14), bool); mask[3:9, 4:10, 5:11] = True
    assert crop_slices(mask) == (slice(1, 11), slice(2, 12), slice(3, 13))
    mask[6, 7, 8] = False
    assert interior_mask(mask)[6, 7, 8]
    assert not interior_mask(mask)[5, 7, 8]


def test_phase_unit_scale_and_step_size():
    assert abs(PHASE_PER_PPM-2*np.pi*42.58*2.89362*.025) < 1e-14
    assert 0 < step_sizes(SPACING) < .1


def test_source_stopping_is_single_update_every_ten():
    rng = np.random.default_rng(8)
    f = rng.normal(size=(7, 8, 9))*.1
    mask = np.ones_like(f, bool)
    r = primal_dual(f, mask, tolerance=2, max_iterations=100)
    assert r['iterations'] == 10
    assert r['history'].shape == (1, 3)


@pytest.mark.parametrize('lam,ratio,iterations', [(-1, 1, 10), (.1, -1, 10), (.1, 1, 0)])
def test_invalid_solver_parameters(lam, ratio, iterations):
    with pytest.raises(ValueError):
        primal_dual(np.zeros((4, 4, 4)), np.ones((4, 4, 4), bool), lam=lam, ratio=ratio, max_iterations=iterations)
