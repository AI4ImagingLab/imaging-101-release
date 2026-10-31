import numpy as np
from src.physics_model import double_gamma_hrf,canonical_basis,glsco,knee_pt,BOLD_event_vector,kernel_at_tr,convolve_periodic,deconvolve_ridge,ideal_filter
def test_basis_and_gamma_normalization():
    h=double_gamma_hrf(2/3,np.array([6,16,1,1,6,0,24]),3)
    assert h.shape==(37,) and np.isfinite(h).all()
    np.testing.assert_allclose(h.sum(),1,atol=1e-15)
    b=canonical_basis();g=b.T@b;np.testing.assert_allclose(g-np.diag(np.diag(g)),0,atol=1e-15)
def test_known_linear_regression():
    x=np.column_stack([np.ones(15),np.arange(15.)]);y=2+3*x[:,1]
    variance,beta=glsco(x,y,AR_lag=0)
    np.testing.assert_allclose(beta,[2,3],rtol=1e-14,atol=1e-14);assert variance<1e-25
def test_known_knee_and_events():
    _,k=knee_pt(np.array([10.,8.,6.,5.,4.,3.]));assert k==2
    y=np.array([0.,0.,3.,0.,0.,0.,4.,0.,0.]);e=BOLD_event_vector(9,y,np.array([1.]),1,[])
    np.testing.assert_array_equal(np.flatnonzero(e.toarray()),[2,6])
def test_constant_filter_mean():
    x=np.full((20,2),5.);np.testing.assert_allclose(ideal_filter(x,2,[.01,.08]),5,atol=1e-14)
def test_convolution_adjoint_and_ridge_normal_equation():
    rng=np.random.default_rng(421);n=30;h=rng.normal(size=(37,1));x=rng.normal(size=(n,1));y=rng.normal(size=(n,1))
    sampled=kernel_at_tr(h)[:,0];padded=np.pad(sampled,(0,n-len(sampled)));C=np.column_stack([np.roll(padded,j) for j in range(n)])
    np.testing.assert_allclose(convolve_periodic(x,h)[:,0],C@x[:,0],rtol=1e-12,atol=1e-12)
    np.testing.assert_allclose(np.vdot(C@x[:,0],y[:,0]),np.vdot(x[:,0],C.T@y[:,0]),rtol=1e-12,atol=1e-12)
    estimate=deconvolve_ridge(y,h)[:,0];lam=.1*np.sum(sampled**2)
    np.testing.assert_allclose(C.T@(C@estimate-y[:,0])+lam*estimate,0,atol=1e-12)
