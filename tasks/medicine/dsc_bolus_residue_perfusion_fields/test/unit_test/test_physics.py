"""Independent small known mathematical/physical cases, not self-generated parity."""
import numpy as np
import pytest
from src.physics_model import convolution_matrix,forward,adjoint
from src.preprocessing import signal_to_concentration
from src.solvers import deconvolve,oscillation_index

@pytest.mark.parametrize('circulant',[False,True])
def test_explicit_scalar_convolution_and_adjoint(circulant):
    a=np.array([0.,0.,1.,4.,2.,.4,0.,0.]);n=len(a);L=2*n if circulant else n
    col=np.zeros(L);col[0]=a[0]
    for i in range(1,n-1):col[i]=(a[i-1]+4*a[i]+a[i+1])/6
    if circulant:col[n-1]=(a[-2]+4*a[-1])/6;col[n]=a[-1]/6
    else:col[n-1]=a[-1]
    r=np.exp(-np.arange(L)/4);direct=np.zeros(n)
    for i in range(n):
        for j in range(L):
            if circulant:direct[i]+=1.55*col[(i-j)%L]*r[j]
            elif i>=j:direct[i]+=1.55*col[i-j]*r[j]
    np.testing.assert_allclose(forward(r,a,circulant=circulant),direct,rtol=2e-15,atol=1e-15)
    y=np.linspace(-1,1,n)
    assert abs(np.dot(forward(r,a,circulant=circulant),y)-np.dot(r,adjoint(y,a,circulant=circulant)))<2e-14

def test_slice_specific_baseline_known_case():
    x=np.full((2,2,3,80),100.,dtype=np.float64);x[1]=700
    x[0,...,11:]=70;x[1,...,7:]=490
    c,s0,counts=signal_to_concentration(x,np.ones(x.shape[:-1],bool))
    np.testing.assert_array_equal(counts,[10,6]);np.testing.assert_allclose(s0[0],100);np.testing.assert_allclose(s0[1],700)
    np.testing.assert_allclose(c[0,...,11:],-np.log(.7)/.025)
    np.testing.assert_allclose(c[1,...,7:],-np.log(.7)/.025)

@pytest.mark.parametrize('method',['svd','csvd','osvd'])
def test_inverse_time_units_and_linear_scale(method):
    a=np.r_[np.zeros(4),np.exp(-np.arange(20)/4),np.zeros(8)]
    y=forward(np.exp(-np.arange(64)/8),a)[None]
    first=deconvolve(y,a,method=method,tr=1)
    second=deconvolve(y,a,method=method,tr=1.55)
    doubled=deconvolve(2*y,a,method=method,tr=1)
    np.testing.assert_allclose(second['residual']*1.55,first['residual'],rtol=2e-13,atol=2e-13)
    np.testing.assert_allclose(doubled['residual'],2*first['residual'],rtol=2e-13,atol=2e-13)

def test_oscillation_index_manual_case():
    r=np.array([[1.,.5,.25,.125]])
    assert oscillation_index(r)[0]==.375/4

def test_invalid_log_is_not_clipped():
    with pytest.raises(ValueError):signal_to_concentration(np.zeros((1,1,1,80)),np.ones((1,1,1),bool))
