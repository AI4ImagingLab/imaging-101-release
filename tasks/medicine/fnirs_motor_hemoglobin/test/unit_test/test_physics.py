import numpy as np
import pytest
from src.preprocessing import optical_density,load_observations,SAMPLE_RATE_HZ,WAVELENGTHS_NM,PATHLENGTH_FACTOR
from src.physics_model import forward,adjoint,absorption_matrix
from src.solvers import invert_density,reconstruct
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]

def test_log_invariance_and_reference():
    x=np.array([[[[1.,2.,4.],[2.,4.,8.]]]])
    np.testing.assert_allclose(optical_density(x),-np.log(x/x.mean(axis=-1,keepdims=True)))
    np.testing.assert_allclose(optical_density(x*47),optical_density(x),atol=1e-15)
    for bad in (0.,-1.,np.nan,np.inf):
        with pytest.raises(ValueError):optical_density(np.array([bad]))

def test_analytic_inverse_units_and_sign():
    e=np.eye(2)[None];d=np.array([[0.01]])
    c=np.array([[[[1e-6,-2e-6],[3e-6,-4e-6]]]])
    od=forward(c,e,d)
    np.testing.assert_allclose(od,c*60.,atol=0,rtol=1e-15)
    np.testing.assert_allclose(invert_density(od,e,d),c,rtol=1e-15,atol=1e-21)
    np.testing.assert_allclose(invert_density(od,e,d*2),c/2,rtol=1e-15,atol=1e-21)

def test_adjoint_and_finite_difference():
    obs=load_observations(ROOT/'data/raw_data.npz');rng=np.random.default_rng(1)
    c=rng.normal(size=(1,68,2,3));y=rng.normal(size=c.shape)
    ac=forward(c,obs['extinction'],obs['distances']);aty=adjoint(y,obs['extinction'],obs['distances'])
    np.testing.assert_allclose(np.sum(ac*y),np.sum(c*aty),rtol=1e-13)

def test_public_constants():
    text=(ROOT/'README.md').read_text(encoding='utf-8')
    for value in (str(SAMPLE_RATE_HZ),'760','850','6.0'):assert value in text
    assert WAVELENGTHS_NM==(760.,850.) and PATHLENGTH_FACTOR==6.
