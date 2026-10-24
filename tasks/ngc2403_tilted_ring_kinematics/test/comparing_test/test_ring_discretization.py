from pathlib import Path
import numpy as np
from src import physics_model as model
F=Path(__file__).resolve().parents[1]/'fixtures'


def test_native_beam_kernel_from_acquisition():
    function=getattr(model,'beam_kernel',None)
    assert callable(function),'beam discretization is missing'
    with np.load(F/'convolution_1.npz',allow_pickle=False) as z:
        np.testing.assert_allclose(function(),z['input_kernel'],rtol=0,atol=1e-12)


def test_original_inner_ring_interpolation_and_coordinates():
    function=getattr(model,'internal_rings',None)
    assert callable(function),'radial cloud interpolation is missing'
    actual,metadata,edges=function([0,30],np.array([120,8,125.60171508789062],dtype=np.float32))
    with np.load(F/'cloud_model_1.npz',allow_pickle=False) as z:
        np.testing.assert_allclose(actual,z['input_internal_rings'],rtol=0,atol=1e-8)
        np.testing.assert_allclose(metadata,z['param_metadata'],rtol=0,atol=1e-8)
