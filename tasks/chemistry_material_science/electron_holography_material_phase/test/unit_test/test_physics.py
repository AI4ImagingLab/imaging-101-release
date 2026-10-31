from pathlib import Path
import numpy as np
import pytest
from src.physics_model import aperture, interference, radial_frequency, sideband_wave
from src.preprocessing import load_observations
from src.solvers import reconstruct

ROOT = Path(__file__).resolve().parents[2]


def test_independent_analytic_interference_phase_sign():
    # Analytic constant object: sideband must be O*conj(R), not its conjugate.
    shape = (64, 64)
    obj = np.full(shape, .7 * np.exp(.8j))
    ref = np.full(shape, 1.3 * np.exp(-.2j))
    carrier = (12, 9)
    measured = interference(obj, ref, carrier)
    result = sideband_wave(measured, carrier, 4, .2)
    np.testing.assert_allclose(result, obj * ref.conj(), atol=2e-14)
    assert abs(np.angle(result[0, 0]) - 1.) < 1e-13


def test_constant_intensity_is_rejected_by_sideband():
    result = sideband_wave(np.ones((64, 64)), (12, 9), 4, .2)
    assert np.linalg.norm(result) < 1e-12


def test_aperture_properties():
    values = aperture(np.array([0., 2., 4.]), 2., .2)
    np.testing.assert_allclose(values, [1, .5, 0], atol=1e-15)
    assert radial_frequency((8, 10))[0, 0] == 0


@pytest.mark.parametrize('radius,smoothness', [(0, 1), (1, 0), (-1, 1)])
def test_invalid_aperture(radius, smoothness):
    with pytest.raises(ValueError):
        aperture(np.array([0.]), radius, smoothness)


def test_data_and_phase_calibration():
    data = load_observations(ROOT / 'data/raw_data.npz')
    result = reconstruct(data['vacuum'], data['vacuum'])
    np.testing.assert_allclose(result, 1, atol=3e-16)
    assert data['hologram'].dtype == np.float64
    assert not np.array_equal(data['hologram'], data['vacuum'])


@pytest.mark.parametrize('mode', ['missing', 'shape', 'nan', 'negative'])
def test_preprocessing_rejects_bad_inputs(tmp_path, mode):
    x = np.ones((1, 512, 512))
    if mode == 'shape': x = x[:, :-1]
    if mode == 'nan': x[0, 0, 0] = np.nan
    if mode == 'negative': x[0, 0, 0] = -1
    arrays = {'hologram': x, 'vacuum': x}
    if mode == 'missing': arrays.pop('vacuum')
    path = tmp_path / 'data.npz'
    np.savez(path, **arrays)
    with pytest.raises(ValueError): load_observations(path)


def test_undefined_vacuum_division():
    zeros = np.zeros((1, 512, 512))
    with pytest.raises(ValueError): reconstruct(zeros, zeros)
