import numpy as np
import pytest
from src.generate_data import add_gaussian_perturbation

def test_framewise_noise_preserves_frozen_pre_refactor_RNG_order_bitwise():
    images=np.arange(16*3*5,dtype=np.float32).reshape(16,3,5)/100
    rms=np.sqrt(np.mean(images.astype(np.float64)**2,axis=(1,2)))
    for fraction in (.005,.01,.02):
        expected=images.astype(np.float64);rng=np.random.default_rng(31917)
        for i in range(16):expected[i]+=rng.normal(0,fraction*rms[i],size=images.shape[1:])
        actual=add_gaussian_perturbation(images,rms,fraction)
        np.testing.assert_array_equal(actual,expected)
        assert actual.dtype==np.float64
    np.testing.assert_array_equal(images,np.arange(16*3*5,dtype=np.float32).reshape(16,3,5)/100)

def test_generation_returns_unclipped_observations_and_zero_fraction_copy():
    images=np.zeros((2,3,4));rms=np.ones(2)
    noisy=add_gaussian_perturbation(images,rms,1)
    assert np.any(noisy<0) and np.any(noisy>0)
    np.testing.assert_array_equal(add_gaussian_perturbation(images,rms,0),images)

@pytest.mark.parametrize('fraction',[-1,np.nan,np.inf])
def test_invalid_sensitivity_fraction_is_rejected(fraction):
    with pytest.raises(ValueError):add_gaussian_perturbation(np.ones((2,3,4)),np.ones(2),fraction)
