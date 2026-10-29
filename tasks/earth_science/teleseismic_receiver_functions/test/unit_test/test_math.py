import numpy as np
from scipy.fft import fft,ifft
from src.physics_model import convolve,adjoint,rotate_zne,filters
from src.solvers import waterlevel
from src.visualization import metrics
def test_convolution_adjoint_and_constant_gain():
    rng=np.random.default_rng(11);s,x,y=rng.normal(size=(3,64))
    np.testing.assert_allclose(np.dot(convolve(s,x,64),y),np.dot(x,adjoint(s,y,64)),rtol=1e-13,atol=1e-12)
    impulse=np.zeros(64);impulse[0]=2;np.testing.assert_allclose(convolve(impulse,x,64),2*x,atol=1e-14)
def test_rotation_orthogonal_units_and_sign():
    x=np.array([[1.,2],[3.,4],[5.,6]])
    np.testing.assert_allclose(rotate_zne(x,0),[x[0],-x[1],-x[2]],atol=1e-15)
    np.testing.assert_allclose(np.sum(rotate_zne(x)**2,axis=0),np.sum(x*x,axis=0),rtol=1e-14)
def test_analytic_source_scaled_response_and_gaussian_delay():
    x=np.zeros((3,1024));x[:,0]=[1,2,-3]
    rf,d=waterlevel(x,waterlevel=.05,gauss=.5,shift=2.)
    np.testing.assert_allclose(rf[1],2*rf[0],atol=1e-14);np.testing.assert_allclose(rf[2],-3*rf[0],atol=1e-14)
    assert np.argmax(rf[0])==40
    g,f=filters(1024,.5,0);idx=np.argmin(abs(f-.5));np.testing.assert_allclose(g[idx].real,np.exp(-.5*(f[idx]/.5)**2))
def test_metrics_known_cases():
    assert metrics(np.array([0.,1.]),np.array([0.,1.]))==dict(ncc=1.,nrmse=0.)
    m=metrics(np.array([1.,0.]),np.array([0.,1.]));assert m['ncc']==0.;assert m['nrmse']==1.
def test_public_constants_and_data():
    from pathlib import Path
    from src.preprocessing import BACK_AZIMUTH,SAMPLING_RATE,load_data
    r=Path(__file__).resolve().parents[2];text=(r/'README.md').read_text(encoding='utf-8')
    assert str(BACK_AZIMUTH) in text and str(SAMPLING_RATE) in text
    assert load_data(r/'data/raw_data.npz').shape==(3,4801)
