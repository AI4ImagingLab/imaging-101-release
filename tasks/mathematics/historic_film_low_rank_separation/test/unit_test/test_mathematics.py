import numpy as np
from src.physics_model import shrink,singular_threshold
from src.solvers import decompose

def test_known_proximal_operators_and_signs():
    np.testing.assert_array_equal(shrink(np.array([-3.,-.2,0.,1.,4.]),1),[-2.,0.,0.,0.,3.])
    np.testing.assert_allclose(singular_threshold(np.diag([3.,1.]),2),np.diag([1.,0.]),atol=1e-14)
    # Orthogonal equivariance follows from the nuclear norm, including signs.
    q=np.array([[0.,-1.],[1.,0.]])
    np.testing.assert_allclose(singular_threshold(q@np.diag([3.,1.]),2),q@np.diag([1.,0.]),atol=1e-14)

def test_nuclear_proximal_optimality_independent_case():
    rng=np.random.default_rng(8);m=rng.normal(size=(11,7));tau=.8
    low=singular_threshold(m,tau)
    # Proximal residual is nuclear subgradient: spectral norm<=1 and
    # inner product with low equals its independently computed nuclear norm.
    sub=(m-low)/tau
    assert np.linalg.norm(sub,2)<=1+1e-12
    np.testing.assert_allclose(np.sum(sub*low),np.linalg.svd(low,compute_uv=False).sum(),rtol=1e-12)

def test_additive_model_and_scale_covariance():
    d=np.diag([4.,2.,1.]);a=decompose(d,max_iter=2000);b=decompose(7*d,max_iter=2000)
    np.testing.assert_allclose(a[0]+a[1],d,atol=1e-6)
    np.testing.assert_allclose(b[0],7*a[0],rtol=1e-10,atol=1e-10)
    np.testing.assert_allclose(b[1],7*a[1],rtol=1e-10,atol=1e-10)


def test_augmented_lagrangian_square_completion_sign_and_threshold():
    # Independent subproblem check of Eq5.1, including mu!=1 and nonzero dual.
    d=np.diag([4.,2.]);s=np.diag([.3,.4]);y=np.diag([.7,-.2]);mu=2.;lam=.6
    low=singular_threshold(d-s+y/mu,1/mu)
    def objective(l):return np.linalg.svd(l,compute_uv=False).sum()+np.sum(y*(d-l-s))+mu/2*np.sum((d-l-s)**2)
    for delta in [-.1,.1]:assert objective(low)<=objective(low+delta*np.eye(2))
    sparse=shrink(d-low+y/mu,lam/mu)
    def objective_s(z):return lam*np.sum(abs(z))+np.sum(y*(d-low-z))+mu/2*np.sum((d-low-z)**2)
    for delta in [-.1,.1]:assert objective_s(sparse)<=objective_s(sparse+delta*np.eye(2))
