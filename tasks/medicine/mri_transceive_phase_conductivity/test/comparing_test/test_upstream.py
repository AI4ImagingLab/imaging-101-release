from pathlib import Path
import numpy as np
import pytest
from src.preprocessing import load_observations, average_repeats
from src.physics_model import second_derivative_kernels, phase_conductivity, full_helmholtz
from src.solvers import reconstruct

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize('kind,shape', list(enumerate(['cross', 'sphere', 'cube'])))
@pytest.mark.parametrize('radius', range(1, 6))
def test_independent_original_kernel(kind, shape, radius):
    with np.load(ROOT/'test/fixtures/upstream_stages.npz') as fixture:
        reference = fixture[f'original_s{kind}_r{radius}_kernel']
    np.testing.assert_allclose(second_derivative_kernels(shape, radius), reference, rtol=5e-10, atol=2e-14)


def test_original_repeat_averages():
    phase, magnitude, _ = load_observations(ROOT/'data/raw_data.npz')
    with np.load(ROOT/'test/fixtures/upstream_stages.npz') as fixture:
        np.testing.assert_array_equal(average_repeats(phase), fixture['original_mean_phase'])
        np.testing.assert_array_equal(average_repeats(magnitude), fixture['original_mean_magnitude'])


@pytest.mark.parametrize('kind,shape,radius', [(0,'cross',2),(1,'sphere',3),(2,'cube',5)])
def test_original_phase_and_magnitude_stages(kind, shape, radius):
    with np.load(ROOT/'test/fixtures/upstream_stages.npz') as fixture:
        phase, magnitude = fixture['original_mean_phase'], fixture['original_mean_magnitude']
    with np.load(ROOT/'notebooks/reference_outputs/upstream_results.npz') as fixture:
        ref = fixture[f'original_s{kind}_r{radius}_phase']
        sigma_ref = fixture[f'original_s{kind}_r{radius}_full_sigma']
        eps_ref = fixture[f'original_s{kind}_r{radius}_full_epsr']
    conductivity, mask = phase_conductivity(phase, shape, radius)
    np.testing.assert_array_equal(mask, np.isfinite(ref))
    np.testing.assert_allclose(conductivity, ref, rtol=5e-10, atol=2e-11, equal_nan=True)
    sigma, epsr, full_mask = full_helmholtz(phase, magnitude, shape, radius)
    np.testing.assert_array_equal(full_mask, np.isfinite(sigma_ref))
    np.testing.assert_allclose(sigma, sigma_ref, rtol=5e-10, atol=2e-11, equal_nan=True)
    np.testing.assert_allclose(epsr, eps_ref, rtol=5e-10, atol=2e-9, equal_nan=True)


def test_full_bound_pipeline():
    phase, _, _ = load_observations(ROOT/'data/raw_data.npz')
    result, mask, _ = reconstruct(phase)
    with np.load(ROOT/'evaluation/baseline_reference.npz') as archive:reference = archive['reconstruction'][0]
    np.testing.assert_allclose(result, reference, rtol=5e-10, atol=2e-11)
    assert mask.sum() == 859 and np.all(result[mask] > 0)
