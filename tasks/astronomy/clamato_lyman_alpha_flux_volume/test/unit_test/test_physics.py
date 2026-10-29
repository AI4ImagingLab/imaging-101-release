import numpy as np
from src.physics_model import kernel_from_q,pairwise_covariance,whitened_product
from src.solvers import conjugate_gradients

def test_gaussian_approximation_bound_and_finite_support():
    q=np.linspace(0,10,100001)
    actual=kernel_from_q(q)
    assert np.max(np.abs(actual-.05*np.exp(-q)))<1.8e-5
    assert actual[0]==.05
    assert np.all(actual[q>=8]==0)

def test_covariance_symmetry_noise_positive_and_adjoint():
    rng=np.random.default_rng(4)
    points=rng.uniform([0,0,0],[8,8,12],(100,3))
    w=rng.uniform(.5,5,100)
    c=pairwise_covariance(points,points)
    a=np.eye(100)+w[:,None]*c*w[None,:]
    np.testing.assert_allclose(a,a.T,atol=1e-15)
    assert np.linalg.eigvalsh(a).min()>.99
    u,v=rng.normal(size=(2,100))
    np.testing.assert_allclose(u@whitened_product(c,w,v),v@whitened_product(c,w,u),rtol=1e-14)

def test_custom_pcg_matches_independent_linear_solve():
    rng=np.random.default_rng(5)
    points=rng.normal(size=(60,3))
    c=pairwise_covariance(points,points)
    w=rng.uniform(1,5,60)
    a=np.eye(60)+w[:,None]*c*w[None,:]
    b=rng.normal(size=60)
    x,info=conjugate_gradients(lambda v:a@v,b,np.diag(a),tol=1e-12,step_r=3)
    assert info['success']
    np.testing.assert_allclose(x,np.linalg.solve(a,b),atol=1e-11,rtol=1e-10)

def test_zero_observations_return_zero_without_undefined_state():
    x,info=conjugate_gradients(lambda v:v,np.zeros(8),np.ones(8))
    assert np.array_equal(x,np.zeros(8))
    assert info['iterations']==0
