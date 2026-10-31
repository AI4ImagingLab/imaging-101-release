from pathlib import Path
import numpy as np
from src.physics_model import transform,inverse,tv_gradient,objective
from src.preprocessing import load_observations
ROOT=Path(__file__).resolve().parents[2]
def test_dct_constant_and_adjoint():
    u=np.ones((3,16,24),dtype=float)*2
    c=transform(u)
    np.testing.assert_allclose(c[...,0,0],16,atol=1e-12)
    c[...,0,0]=0
    np.testing.assert_allclose(c,0,atol=1e-12)
    rng=np.random.default_rng(402)
    x=rng.normal(size=u.shape);y=rng.normal(size=(3,2,3,8,8))
    np.testing.assert_allclose(np.vdot(transform(x),y),np.vdot(x,inverse(y)),rtol=1e-10)
    np.testing.assert_allclose(inverse(transform(x)),x,rtol=1e-10,atol=1e-12)
def test_joint_tv_gradient_independent_finite_difference():
    rng=np.random.default_rng(403);x=rng.normal(size=(3,5,7));direction=rng.normal(size=x.shape)
    def tv(u):
        a=np.zeros_like(u);b=a.copy();a[:,:,:-1]=np.diff(u,axis=2);b[:,:-1]=np.diff(u,axis=1)
        return np.sum(np.sqrt(np.sum(a*a+b*b,axis=0)))/np.sqrt(3)
    numerical=(tv(x+1e-6*direction)-tv(x-1e-6*direction))/2e-6
    np.testing.assert_allclose(np.sum(tv_gradient(x)*direction),numerical,rtol=1e-5)
    np.testing.assert_allclose(tv_gradient(np.ones_like(x)),0,atol=0)
def test_exact_public_arrays():
    a,q=load_observations(ROOT/'data/raw_data.npz')
    assert a.dtype==np.int16 and q.dtype==np.uint16
    assert (q>0).all()
    with np.load(ROOT/'test/fixtures/upstream_states.npz') as z:
        np.testing.assert_array_equal(a,z['input_coefficients'])
        np.testing.assert_array_equal(q,z['input_quantization'])
def test_complete_objective_gradient():
    rng=np.random.default_rng(405)
    u=rng.normal(size=(3,8,8));d=rng.normal(size=u.shape)
    q=np.ones((3,8,8),dtype=float)*3;a=np.zeros((3,1,1,8,8),dtype=np.int16)
    alpha=.001*2*255*np.sqrt(2)
    gradient=tv_gradient(u)+alpha*inverse(transform(u)/(q[:,None,None]**2))
    finite=(objective(u+1e-6*d,a,q)-objective(u-1e-6*d,a,q))/2e-6
    np.testing.assert_allclose(np.sum(gradient*d),finite,rtol=1e-5)
