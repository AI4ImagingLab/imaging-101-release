import numpy as np
from src.physics_model import *

def test_adjoint_and_gradient():
    rng=np.random.default_rng(82)
    shape=(7,9);a=rng.normal(size=(13,63));x=rng.normal(size=shape);y=rng.normal(size=13)
    np.testing.assert_allclose(np.dot(forward(a,x),y),np.sum(x*adjoint(a,y,shape)),rtol=1e-13)
    d=gradient_matrix(shape,sampling=.2)
    np.testing.assert_array_equal(d@np.ones(63),np.zeros(126))
    z=np.indices(shape)[0]*.2;r=np.indices(shape)[1]*.2
    dz=(d@z.ravel()).reshape(2,*shape);dr=(d@r.ravel()).reshape(2,*shape)
    np.testing.assert_allclose(dz[0,:-1],1,atol=1e-14)
    np.testing.assert_allclose(dr[1,:,:-1],1,atol=1e-14)
    w=diffusion_tensor(rng.normal(size=shape),sampling=.2)
    eigen=np.linalg.eigvalsh(np.moveaxis(w,(0,1),(-2,-1)))
    np.testing.assert_allclose(eigen,np.broadcast_to([.01,1],eigen.shape),atol=1e-14)
    lap=diffusion_matrix(w,d)
    direction=rng.normal(size=shape);eps=1e-6
    loss,g=loss_gradient(x,a,y,lap)
    numerical=(loss_gradient(x+eps*direction,a,y,lap)[0]-loss_gradient(x-eps*direction,a,y,lap)[0])/(2*eps)
    np.testing.assert_allclose(numerical,np.sum(g*direction),rtol=1e-7)
    assert loss>=0

def test_toroidal_quadrature():
    from src.preprocessing import R,DR,DZ
    x=np.ones((120,41))
    expected=120*DZ*np.pi*((R[-1]+DR/2)**2-(R[0]-DR/2)**2)
    np.testing.assert_allclose(integrate_power(x),expected,rtol=1e-14)
