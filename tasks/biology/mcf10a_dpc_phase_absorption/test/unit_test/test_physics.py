import numpy as np
from src.physics_model import transfer_functions,forward,adjoint
from src.solvers import normal_matrix,invert,third_difference,shrink
def test_real_adjoint_and_phase_dc_nullspace():
    m=transfer_functions((64,64));rng=np.random.default_rng(510);u=rng.normal(size=(64,64));p=rng.normal(size=(64,64));y=rng.normal(size=(4,64,64));hu,hp=m['Hu'],m['Hp']
    a,b=adjoint(y,hu,hp);assert np.isclose(np.vdot(forward(u,p,hu,hp),y),np.vdot(u,a)+np.vdot(p,b),rtol=1e-12)
    # WOTF construction includes original complex64 source FFT; numerical DC
    # leakage is ~3e-12 at this grid, relative to order-one transfer response.
    np.testing.assert_allclose(forward(np.zeros_like(u),np.ones_like(p),hu,hp),0,atol=1e-10)
def test_weak_absorber_uniform_source_response():
    m=transfer_functions((64,64));out=forward(np.ones((64,64))*.01,np.zeros((64,64)),m['Hu'],m['Hp'])
    np.testing.assert_allclose(out,.02,rtol=1e-6)
def test_third_derivative_fourier_symbol_and_shrink():
    rng=np.random.default_rng(511);x=rng.normal(size=(32,32));d=np.zeros((32,32));d[0,0]=3;d[0,-1]=-1;d[0,1]=-3;d[0,2]=1
    np.testing.assert_allclose(third_difference(x,1),np.fft.ifft2(np.fft.fft2(x)*np.fft.fft2(d)).real,atol=1e-13)
    assert np.max(abs(shrink(np.zeros((2,4,4)),1)))==0
    np.testing.assert_allclose(shrink(np.array([3.,4.])[:,None,None],2).ravel(),[1.8,2.4])
