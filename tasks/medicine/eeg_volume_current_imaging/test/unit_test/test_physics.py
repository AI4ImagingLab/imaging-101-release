import numpy as np
from src.physics_model import forward,adjoint,projection
from src.solvers import reconstruct
def test_adjoint_reference_and_units():
    rng=np.random.default_rng(48);g=rng.normal(size=(8,15));j=rng.normal(size=(5,3,9));v=rng.normal(size=(8,9))
    assert np.isclose(np.vdot(forward(j,g),v),np.vdot(j,adjoint(v,g)),rtol=1e-13)
    assert np.max(abs(forward(j,g).mean(axis=0)))<1e-14
    np.testing.assert_allclose(forward(j*1e-9,g),forward(j,g)*1e-9,rtol=1e-12,atol=1e-24)
def test_analytic_svd_ridge_and_stationarity():
    rng=np.random.default_rng(49);g=rng.normal(size=(8,15));y=rng.normal(size=(8,4));out,stage=reconstruct({'lead_field':g,'measurements':y})
    p=projection(8);a=p@g/1e-5;x=out['reconstruction'][0].reshape(15,4);b=p@y/1e-5
    alpha=stage['trace_scale']/9
    np.testing.assert_allclose((a.T@a+alpha*np.eye(15))@x,a.T@b,rtol=1e-10,atol=1e-3)
def test_projector_symmetry_idempotence_constant_rejection():
    p=projection();np.testing.assert_allclose(p@p,p,atol=1e-14);assert np.max(abs(p@np.ones(64)))<1e-14
