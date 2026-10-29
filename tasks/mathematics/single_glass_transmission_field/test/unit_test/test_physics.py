import numpy as np
from src.physics_model import gradient,divergence,laplacian,threshold_gradients
from src.solvers import reconstruct
def test_adjoint_neumann_boundary():
 rng=np.random.default_rng(26);x=rng.normal(size=(7,9,3));q=rng.normal(size=(7,9,3,2));q[:,-1,:,0]=0;q[-1,:,:,1]=0
 np.testing.assert_allclose(np.sum(gradient(x)*q),-np.sum(x*divergence(q)),atol=1e-12)
 np.testing.assert_allclose(np.sum(x*laplacian(x)),-np.sum(gradient(x)**2),atol=1e-12)
 np.testing.assert_allclose(laplacian(x).sum(axis=(0,1)),0,atol=1e-12)
def test_cosine_and_constant_physical_cases():
 y,x=np.indices((8,11));known=.4+.1*np.cos(np.pi*(y+.5)*2/8)*np.cos(np.pi*(x+.5)*3/11)
 out=reconstruct(known,h=0);np.testing.assert_allclose(out,known,atol=1e-7,rtol=0)
 np.testing.assert_allclose(reconstruct(np.full((8,11),.37)),.37,atol=1e-8,rtol=0)
def test_strict_hard_threshold_tie():
 x=np.array([[0.,1.],[0.,1.]])
 np.testing.assert_array_equal(threshold_gradients(x,1.),np.zeros((2,2,2)))
 assert np.any(threshold_gradients(x,np.nextafter(1.,0.))!=0)
