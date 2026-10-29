import numpy as np
from scipy.sparse import vstack
from src.physics_model import derivative_matrices,system,weights
from src.solvers import integrate

def test_hand_derived_direction_signs_and_constant_nullspace():
    mask=np.ones((3,4),bool);n=-np.ones(mask.sum());right,left,up,down=derivative_matrices(mask,n,n)
    row,col=np.indices(mask.shape);z=(2*row+3*col).ravel()
    np.testing.assert_allclose((right@z).reshape(mask.shape)[:,:-1],-3)
    np.testing.assert_allclose((left@z).reshape(mask.shape)[:,1:],-3)
    np.testing.assert_allclose((up@z).reshape(mask.shape)[1:,:],2)
    np.testing.assert_allclose((down@z).reshape(mask.shape)[:-1,:],2)
    for a in [right,left,up,down]:np.testing.assert_array_equal(a@np.ones(mask.sum()),0)

def test_adjoint_and_objective_gradient_with_mask_boundary():
    rng=np.random.default_rng(41);mask=np.ones((7,9),bool);mask[2:4,3:6]=False
    normal=rng.normal(size=mask.shape+(3,));a,b,ops=system(normal,mask)
    z=rng.normal(size=mask.sum());q=rng.normal(size=a.shape[0]);d=rng.normal(size=z.size);w=rng.uniform(.1,1,len(q))
    np.testing.assert_allclose((a@z)@q,z@(a.T@q),rtol=1e-13)
    eps=1e-6
    loss=lambda x:np.sum(w*(a@x-b)**2)
    fd=(loss(z+eps*d)-loss(z-eps*d))/(2*eps)
    np.testing.assert_allclose(fd,2*d@(a.T@(w*(a@z-b))),rtol=1e-8)

def test_analytic_plane_gauge_and_units():
    row,col=np.indices((13,15));truth=.15*row+.25*col;truth-=truth.mean()
    scale=np.sqrt(1+.15**2+.25**2);n=np.empty(truth.shape+(3,));n[:]=[.25/scale,-.15/scale,1/scale]
    z,diag=integrate(n,np.ones(truth.shape,bool),k=0,cg_tol=1e-11)
    np.testing.assert_allclose(z,truth,rtol=1e-8,atol=1e-8)
    assert abs(z.mean())<1e-12 and all(v==0 for v in diag['cg_info'])
    # A doubled physical step scales heights, not input normal directions.
    z2,_=integrate(n,np.ones(truth.shape,bool),k=0,step_size=2,cg_tol=1e-11)
    np.testing.assert_allclose(z2,2*truth,rtol=1e-8,atol=1e-8)

def test_complementary_weights_and_flat_case():
    mask=np.ones((6,7),bool);n=np.zeros(mask.shape+(3,));n[...,2]=1
    a,b,ops=system(n,mask);w,h,v=weights(np.zeros(mask.sum()),ops,4)
    np.testing.assert_array_equal(w,.5);assert len(w)==4*mask.sum()
    z,diag=integrate(n,mask,k=4);assert np.isfinite(diag['energy']).all();np.testing.assert_array_equal(z,0)
