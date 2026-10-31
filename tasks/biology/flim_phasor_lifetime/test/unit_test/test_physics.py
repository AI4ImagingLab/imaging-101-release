import numpy as np
from src.physics_model import normalized_phasor,single_exponential_phasor,calibrate,lifetimes,FREQUENCY_MHZ,LASER_PERIOD_NS
def test_delta_harmonic_sign_and_counts():
    x=np.zeros(256);x[64]=10
    m,g,s=normalized_phasor(x)
    np.testing.assert_allclose([m,g,s],[10/256,0,1],atol=1e-15)
def test_circular_convolution_transfer():
    x=np.exp(-np.arange(256)/30);irf=np.zeros(256);irf[5:9]=[1,3,2,1]
    h=np.fft.ifft(np.fft.fft(x)*np.fft.fft(irf)).real
    _,g,s=normalized_phasor(x);_,a,b=normalized_phasor(irf);_,G,S=normalized_phasor(h)
    np.testing.assert_allclose(G+1j*S,(g+1j*s)*(a+1j*b),rtol=1e-13)
    assert np.isclose(h.sum(),x.sum()*irf.sum())
def test_apparent_lifetime_analytic_identity():
    tau=np.array([.2,1.,4.1,10.])
    g,s=single_exponential_phasor(tau)
    phase,mod=lifetimes(g,s)
    np.testing.assert_allclose(phase,tau,rtol=1e-13)
    np.testing.assert_allclose(mod,tau,rtol=1e-13)
def test_calibration_standard_maps_to_known_lifetime():
    calibration=np.exp(-np.arange(256)/40)[None,None,:]
    _,g,s=normalized_phasor(calibration)
    G,S=calibrate(g,s,calibration)
    phase,mod=lifetimes(G,S)
    np.testing.assert_allclose(phase,4.1,rtol=1e-13);np.testing.assert_allclose(mod,4.1,rtol=1e-13)
def test_metadata_and_no_spatial_binning():
    from pathlib import Path
    root=Path(__file__).resolve().parents[2]
    text=(root/'README.md').read_text(encoding='utf-8')
    assert str(FREQUENCY_MHZ) in text and str(LASER_PERIOD_NS) in text
    with np.load(root/'data/raw_data.npz') as f:
        assert f['measurements'].shape==(1,256,256,256)
def test_visualization_metrics():
    from src.visualization import metrics
    a=np.array([0.,1.,2.]);b=2*a
    n,e=metrics(a,b)
    assert abs(n-1)<1e-14;assert abs(e-np.sqrt(5/3)/4)<1e-14
