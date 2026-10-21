"""Frozen oracle parity and independent analytic/physical checks."""
from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import quad

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def oracle():
    with np.load(ROOT / 'test/fixtures/upstream_stages.npz', allow_pickle=False) as data:
        yield data


def test_preprocessing_preserves_observations(oracle):
    from src.preprocessing import load_projection
    assert np.array_equal(load_projection(ROOT/'data/raw_data.npz')[0], oracle['input_projection'])


def test_intermediate_coefficients_and_derivative(oracle):
    from src.physics_model import state_coefficients
    from src.solvers import inverse_drive
    phi, b0, b1 = state_coefficients(257, 'inverse', 1)
    for actual, key in [(phi, 'phi'), (b0, 'B0'), (b1, 'B1')]:
        np.testing.assert_allclose(actual, oracle['output_'+key], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(inverse_drive(oracle['input_projection'], 0.025, 1),
                               oracle['output_drive'], rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize('order', [0, 1])
def test_bound_forward_inverse(oracle, order):
    from src.physics_model import forward_transform
    from src.solvers import inverse_transform
    inverse_key = 'output_reconstruction' if order else 'output_inverse_zoh'
    forward_key = 'output_forward' if order else 'output_forward_zoh'
    np.testing.assert_allclose(inverse_transform(oracle['input_projection'], 0.025, order),
                               oracle[inverse_key], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(forward_transform(oracle['input_emission'], 0.025, order),
                               oracle[forward_key], rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize('direction', ['forward', 'inverse'])
@pytest.mark.parametrize('order', [0, 1])
def test_signed_probe_and_one_dimensional_input(oracle, direction, order):
    from src.physics_model import forward_transform
    from src.solvers import inverse_transform
    function = forward_transform if direction == 'forward' else inverse_transform
    expected = oracle[f'output_probe_{direction}_{order}']
    np.testing.assert_allclose(function(oracle['input_probe'], 0.13, order), expected,
                               rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(function(oracle['input_probe'][0], 0.13, order), expected[0],
                               rtol=1e-12, atol=1e-12)


def test_gaussian_pair_absolute_scale_and_grid():
    from src.physics_model import forward_transform
    from src.solvers import inverse_transform
    dr = 0.025
    r = np.arange(257)*dr
    emission = np.exp(-(r/1.1)**2)
    projection = np.sqrt(np.pi)*1.1*emission
    for actual, expected in [(forward_transform(emission, dr, 1), projection),
                             (inverse_transform(projection, dr, 1), emission)]:
        assert np.linalg.norm(actual-expected)/np.linalg.norm(expected) < 0.015


def test_generator_line_of_sight_quadrature_and_noise():
    from src.generate_data import synthetic_fields
    from src.preprocessing import DR, DZ, NOISE_STD
    truth, projection, clean = synthetic_fields(seed=601)
    r = np.arange(257)*DR
    z = (np.arange(65)-32)*DZ
    assert truth.shape == projection.shape == clean.shape == (1, 65, 257)
    assert np.min(truth) >= 0
    for row, col in [(0, 0), (32, 40), (43, 11), (64, 100)]:
        zz, yy = z[row], r[col]
        def field(x):
            rr2 = x*x + yy*yy
            return (0.9*np.exp(-(zz/2.3)**2)*np.exp(-rr2/1.5**2)
                    -0.65*np.exp(-(zz/2.3)**2)*np.exp(-rr2/0.7**2)
                    +0.5*np.exp(-((zz-1.1)/0.65)**2)*np.exp(-rr2/0.42**2))
        integral = 2*quad(field, 0, np.inf, epsabs=1e-10)[0]
        assert abs(clean[0, row, col]-integral) < 1e-6
    residual = projection-clean
    assert abs(float(residual.mean())) < NOISE_STD*0.04
    assert 0.96 < float(np.mean((residual/NOISE_STD)**2)) < 1.04


def test_generator_forward_roundtrip_and_flux(oracle):
    from src.generate_data import synthetic_fields
    from src.physics_model import forward_transform
    truth, _, clean = synthetic_fields(seed=917)
    # Deterministic physical arrays, independent of the random observation draw.
    assert np.array_equal(truth[0], oracle['input_emission'])
    projected = forward_transform(truth[0], 0.025, 1)
    assert np.linalg.norm(projected-clean[0])/np.linalg.norm(clean) < 0.015
    radius = np.arange(257)*0.025
    # Integrated projection equals the volume emission per axial slice.
    projection_flux = 2*np.trapezoid(clean[0], radius, axis=-1)
    emission_flux = 2*np.pi*np.trapezoid(truth[0]*radius[None, :], radius, axis=-1)
    np.testing.assert_allclose(projection_flux, emission_flux, rtol=0.001)


def test_linearity_and_units():
    from src.physics_model import forward_transform
    from src.solvers import inverse_transform
    rng = np.random.default_rng(43)
    a, b = rng.normal(size=(2, 4, 31))
    for fn, scale in [(forward_transform, 2.0), (inverse_transform, 0.5)]:
        np.testing.assert_allclose(fn(2*a-3*b, 0.1, 1), 2*fn(a, 0.1, 1)-3*fn(b, 0.1, 1), atol=1e-12)
        np.testing.assert_allclose(fn(a, 0.2, 1), scale*fn(a, 0.1, 1), atol=1e-12)
        out = fn(a, 0.1, 1)
        assert np.array_equal(out[:, 0], out[:, 1])
        assert np.array_equal(out[:, -1], out[:, -2])


def test_fresh_pipeline_matches_upstream(tmp_path, oracle):
    from src.main import run
    destination = tmp_path/'reconstruction.npz'
    run(ROOT/'data/raw_data.npz', destination)
    with np.load(destination, allow_pickle=False) as result:
        assert result['reconstruction'].shape == (1, 65, 257)
        assert result['reconstruction'].dtype == np.float64
        np.testing.assert_allclose(result['reconstruction'][0], oracle['output_reconstruction'], rtol=1e-12, atol=1e-12)
