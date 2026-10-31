"""Independent qualified-original captures; tolerances frozen before extraction."""
from pathlib import Path
import numpy as np
import pytest
from src import physics_model as model
F=Path(__file__).resolve().parents[1]/'fixtures'

@pytest.mark.parametrize('number',[1,2])
def test_original_beam_convolution(number):
    function=getattr(model,'beam_convolve',None)
    assert callable(function),'local beam convolution is not implemented'
    with np.load(F/f'convolution_{number}.npz',allow_pickle=False) as case:
        actual=function(case['input_cube'],case['input_kernel'],float(case['param_scale']))
        np.testing.assert_allclose(actual,case['output_cube'],rtol=0,atol=1e-6)

@pytest.mark.parametrize('number',[1,2,3])
def test_original_azimuthal_normalization_and_loss(number):
    function=getattr(model,'azimuthal_loss',None)
    assert callable(function),'local azimuthal loss is not implemented'
    with np.load(F/f'loss_{number}.npz',allow_pickle=False) as case:
        meta=case['param_metadata']
        cube,summary=function(case['input_cube'],case['input_observation'],case['input_mask'],case['input_rings'],meta)
        np.testing.assert_allclose(cube,case['output_cube'],rtol=0,atol=1e-6)
        np.testing.assert_allclose(summary,case['output_summary'],rtol=0,atol=1e-6)

@pytest.mark.parametrize('number',[1,2])
def test_original_conditional_cloud_operator(number):
    function=getattr(model,'cloud_cube',None)
    assert callable(function),'local conditional cloud acquisition is not implemented'
    with np.load(F/f'cloud_model_{number}.npz',allow_pickle=False) as case:
        actual=function(case['input_internal_rings'],case['param_metadata'],quadrature=case['input_quadrature'])
        np.testing.assert_allclose(actual,case['output_cube'],rtol=0,atol=1e-4)
        samples=model._conditioned_cloud_samples(case['input_internal_rings'],case['param_metadata'],case['input_quadrature'])
        projection=np.concatenate([np.column_stack((np.full(len(profile),ir),profile,azimuth)) for ir,(profile,azimuth,_,_) in enumerate(samples)])
        np.testing.assert_allclose(projection,case['output_cloud_projection'],rtol=0,atol=1e-12)
