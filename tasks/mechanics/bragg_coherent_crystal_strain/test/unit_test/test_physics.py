import numpy as np
from scipy.signal import convolve
from src import physics_model as physics
from src.preprocessing import reciprocal_geometry, real_space_basis, load_observations
from src.solvers import align_complex


def test_fft_inverse_and_adjoint_scale():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(9, 8, 7)) + 1j * rng.normal(size=(9, 8, 7))
    y = rng.normal(size=x.shape) + 1j * rng.normal(size=x.shape)
    np.testing.assert_allclose(physics.direct_wavefield(physics.reciprocal_wavefield(x)), x, atol=1e-13)
    np.testing.assert_allclose(np.vdot(physics.reciprocal_wavefield(x), y),
                               np.vdot(x, physics.direct_wavefield(y)) * x.size, rtol=1e-12)


def test_linear_convolution_odd_even_against_direct_sum():
    rng = np.random.default_rng(3)
    for shape in [(3, 3, 3), (2, 4, 2)]:
        x = rng.uniform(size=(6, 7, 8))
        k = rng.uniform(size=shape)
        np.testing.assert_allclose(physics.linear_convolution(x, k),
                                   convolve(x, k, mode="same", method="direct"), rtol=1e-12, atol=1e-12)


def test_finite_window_kernel_adjoint_and_sensitivity_against_dense_matrix():
    rng = np.random.default_rng(41)
    for kshape in [(3, 3, 3), (4, 3, 2), (4, 4, 4)]:
        rshape = (5, 6, 7)
        coherent = rng.uniform(.1, 2, rshape)
        origin = tuple((n - 1) // 2 for n in kshape)
        matrix = np.zeros((np.prod(rshape), np.prod(kshape)))
        for row, index in enumerate(np.ndindex(rshape)):
            for column, offset in enumerate(np.ndindex(kshape)):
                source = tuple(i + s - u for i, s, u in zip(index, origin, offset))
                if all(0 <= i < n for i, n in zip(source, rshape)):
                    matrix[row, column] = coherent[source]
        residual = rng.normal(size=rshape)
        result = physics.kernel_adjoint(residual, coherent, kshape)
        sensitivity = physics.kernel_adjoint(np.ones(rshape), coherent, kshape)
        np.testing.assert_allclose(result.ravel(), matrix.T @ residual.ravel(), atol=2e-13)
        np.testing.assert_allclose(sensitivity.ravel(), matrix.sum(axis=0), atol=2e-13)


def test_centered_frequency_impulse_blur_keeps_full_local_flux():
    shape = (121, 18, 19)
    wavefield = np.zeros(shape)
    center = tuple(n // 2 for n in shape)
    wavefield[center] = 1.
    kernel = np.ones((7, 7, 7)) / 343
    obj = physics.direct_wavefield(wavefield)
    prediction = physics.observation_intensity(obj, kernel)
    np.testing.assert_allclose(prediction.sum(), 1., atol=1e-13)
    np.testing.assert_allclose(prediction[center], 1 / 343, atol=1e-15)
    assert (prediction > 1e-12).sum() == 343


def test_centered_odd_axis_fourier_peak_and_roundtrip():
    shape = (121, 18, 19)
    obj = np.ones(shape, dtype=np.complex128)
    wavefield = physics.reciprocal_wavefield(obj)
    assert np.unravel_index(abs(wavefield).argmax(), shape) == tuple(n // 2 for n in shape)
    np.testing.assert_allclose(physics.direct_wavefield(wavefield), obj, atol=1e-13)


def test_real_reciprocal_dual_basis():
    b, _ = reciprocal_geometry()
    d = real_space_basis()
    np.testing.assert_allclose(b.T @ d, 2 * np.pi * np.diag(1 / np.array([126, 256, 256])),
                               atol=1e-15)


def test_complete_real_observation_shapes():
    from pathlib import Path
    root = Path(__file__).parents[2]
    counts, theta = load_observations(root / "data/raw_data.npz")
    assert counts.shape == (2, 121, 256, 256)
    assert [int(a.sum()) for a in counts] == [7601377, 6558518]
    assert np.all(np.diff(theta, axis=-1) > 0)


def test_alignment_recovers_global_phase_translation_and_twin():
    rng = np.random.default_rng(5)
    original = rng.normal(size=(9, 8, 7)) + 1j * rng.normal(size=(9, 8, 7))
    for twin in [False, True]:
        candidate = np.conj(np.flip(original)) if twin else original.copy()
        candidate = np.roll(candidate, (3, -2, 1), axis=(0, 1, 2)) * np.exp(.7j)
        aligned, _ = align_complex(original, candidate)
        np.testing.assert_allclose(aligned, original, rtol=1e-12, atol=1e-12)


def test_projected_strain_known_plane_wave_and_gauge_removal():
    shape = (13, 12, 11)
    grid = np.indices(shape)
    phase = .02 * grid[0] + .03 * grid[1] - .01 * grid[2]
    obj = np.exp(1j * phase)
    strain, mask = physics.projected_strain(obj, np.eye(3), np.array([1., 2., 3.]))
    assert mask.sum() > 0
    np.testing.assert_allclose(strain[mask], 0., atol=1e-15)


def test_nonuniform_strain_scale_and_sign_on_oblique_basis():
    shape = (11,10,9)
    basis = np.array([[1.,.2,.1],[.1,.8,-.1],[.05,.3,1.2]])
    coordinate = np.einsum('ij,j...->i...',basis,np.indices(shape))
    hessian = np.array([[2e-4,1e-4,0.],[1e-4,-3e-4,5e-5],[0.,5e-5,1e-4]])
    slope = np.array([.01,-.02,.015])
    phase = .5*np.einsum('i...,ij,j...->...',coordinate,hessian,coordinate)
    phase += np.einsum('i,i...->...',slope,coordinate)
    q = np.array([.3,.7,-.4])
    strain,mask = physics.projected_strain(np.exp(1j*phase),basis,q)
    gradient = np.einsum('ij,j...->i...',hessian,coordinate)+slope[:,None,None,None]
    analytic = np.einsum('i,i...->...',q,gradient)/np.dot(q,q)
    analytic -= analytic[mask].mean()
    assert analytic[mask].min() < 0 < analytic[mask].max()
    np.testing.assert_allclose(strain[mask],analytic[mask],rtol=1e-11,atol=1e-14)
