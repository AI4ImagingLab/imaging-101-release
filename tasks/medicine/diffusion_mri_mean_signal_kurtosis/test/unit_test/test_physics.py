import numpy as np
from src.physics_model import predict,design_matrix
from src.solvers import fit_shells
from src.preprocessing import shell_average,signal_mask

def test_analytic_cumulant_signal_recovery():
    b=np.arange(0,3001,200);D=np.array([.0008]);K=np.array([.7]);S=np.array([200.])
    mean=predict(D,K,S,b);d,k,s=fit_shells(mean,np.ones(16)*33,b,np.array([True]))
    np.testing.assert_allclose(d,D,rtol=2e-8);np.testing.assert_allclose(k,K,rtol=2e-8);np.testing.assert_allclose(s,S,rtol=2e-8)

def test_signal_gain_invariance_and_baseline_scale():
    b=np.arange(0,3001,200);m=predict(np.array([.001]),np.array([.5]),np.array([200.]),b)
    a=fit_shells(m,np.ones(16),b,np.array([True]));c=fit_shells(2*m,np.ones(16),b,np.array([True]))
    np.testing.assert_allclose(a[:2],c[:2],rtol=1e-7);np.testing.assert_allclose(c[2],2*a[2],rtol=1e-7)

def test_powder_average_direction_permutation_and_mask():
    b=np.array([0,1000,1000,2000,2000]);x=np.array([[100,60,70,30,40.]])
    a=shell_average(x,b);j=np.array([4,2,0,3,1]);c=shell_average(x[:,j],b[j])
    for v,w in zip(a,c):np.testing.assert_array_equal(v,w)
    assert signal_mask(x,b).item()
    assert not signal_mask(x*.1,b).item()

def test_zero_b_prediction_and_gradient():
    b=np.array([0.,1000.]);D=np.array([.001]);K=np.array([.7]);S=np.array([100.]);h=1e-8
    y=predict(D,K,S,b);assert y[0,0]==100
    derivative=(predict(D+h,K,S,b)-predict(D-h,K,S,b))/(2*h)
    np.testing.assert_allclose(derivative,y*(-b+b*b*D[:,None]*K[:,None]/3),rtol=1e-8)

def test_gaussian_comparison_baseline_known_case():
    from src.visualization import gaussian_fit
    b=np.arange(0,3001,200);D=np.array([.001]);S=np.array([200.]);y=S[:,None]*np.exp(-D[:,None]*b)
    d,s=gaussian_fit(y,np.ones_like(b)*33,b,np.array([True]))
    np.testing.assert_allclose(d,D,rtol=1e-9);np.testing.assert_allclose(s,S,rtol=1e-9)
