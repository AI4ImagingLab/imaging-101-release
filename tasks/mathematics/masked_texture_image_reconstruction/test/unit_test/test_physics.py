import numpy as np
import pytest
from src.physics_model import *
from src.solvers import project_ball,reconstruct
from src.generate_data import generate_observations

def test_sampling_self_adjoint_and_nullspace():
    rng=np.random.default_rng(77);x=rng.normal(size=(8,9));y=rng.normal(size=(8,9));m=rng.random((8,9))>.6
    assert np.vdot(acquisition(x,m),y)==pytest.approx(np.vdot(x,acquisition(y,m)))
    assert np.any(~m)
    missing=np.where(m,0,rng.normal(size=m.shape))
    np.testing.assert_array_equal(acquisition(x+missing,m),acquisition(x,m))
@pytest.mark.parametrize('degree',[1,2,3])
def test_weighted_physical_adjoints(degree):
    rng=np.random.default_rng(degree);shape=(7,8)
    if degree==1:
        a=rng.normal(size=shape);b=rng.normal(size=shape+(2,))
        lhs=np.sum(gradient(a)*b);rhs=np.sum(a*gradient_adjoint(b))
    elif degree==2:
        a=rng.normal(size=shape+(2,));b=rng.normal(size=shape+(3,))
        lhs=np.sum(symgrad(a)*b*np.array([1,1,2]));rhs=np.sum(a*symgrad_adjoint(b))
    else:
        a=rng.normal(size=shape+(3,));b=rng.normal(size=shape+(4,))
        lhs=np.sum(symgrad2(a)*b*np.array([1,1,3,3]));rhs=np.sum(a*symgrad2_adjoint(b)*np.array([1,1,2]))
    assert lhs==pytest.approx(rhs,abs=1e-12)
def test_affine_field_interior_and_constant_nullspace():
    r,c=np.indices((8,9));u=2*r+3*c+4
    g=gradient(u);np.testing.assert_array_equal(g[:-1,:-1],np.broadcast_to([2,3],(7,8,2)))
    np.testing.assert_array_equal(gradient(np.ones((8,9))),0.)
    np.testing.assert_allclose(symgrad(g)[1:-1,1:-1],0.)
def test_tensor_projection_geometry():
    x=np.array([[[3.,4.,5.]]]);p=project_ball(x,2.,[1,1,2])
    assert np.sqrt(np.sum(p*p*np.array([1,1,2])))==pytest.approx(2.)
    np.testing.assert_allclose(p/x,np.full_like(x,2/np.sqrt(75.)))
def test_independent_generation_no_target_leak():
    rgb=np.array([[[255,0,0],[0,255,0]],[[0,0,255],[255,255,255]]],dtype=np.uint8)
    mask=np.array([[1,0],[1,1]])
    public,truth=generate_observations(rgb,mask)
    assert set(public)=={'observations','mask'}
    np.testing.assert_allclose(truth['reconstruction'],[[[.3,.59],[.11,1.]]])
    assert public['observations'][0,0,1]==0
    np.testing.assert_array_equal(public['mask'],mask[None].astype(bool))
@pytest.mark.parametrize('order',[1,2,3])
def test_hard_observation_constraint(order):
    y=np.arange(48).reshape(6,8)/50.;m=np.indices(y.shape).sum(axis=0)%3==0
    r=reconstruct(np.where(m,y,0),m,order=order,niter=9)
    np.testing.assert_array_equal(r['u'][m],y[m])
@pytest.mark.parametrize('kwargs',[{'order':0},{'niter':0},{'alpha1':-1},{'alpha2':np.inf}])
def test_invalid_algorithm_parameters(kwargs):
    with pytest.raises(ValueError):reconstruct(np.ones((3,4)),np.ones((3,4),bool),**kwargs)

