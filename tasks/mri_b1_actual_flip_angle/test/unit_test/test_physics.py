from pathlib import Path
import numpy as np
from src.physics_model import afi_ratio,invert_afi,exact_steady_state,NOMINAL_FLIP_ANGLE_DEG,TR1_MS,TR2_MS
ROOT=Path(__file__).resolve().parents[2]
def test_analytic_inverse():
    b=np.array([.5,.8,1.,1.3]);ratio=afi_ratio(b)
    signal=np.stack([np.ones_like(b),ratio],axis=-1)
    np.testing.assert_allclose(invert_afi(signal),b,rtol=1e-13,atol=1e-14)
def test_receive_intensity_cancels():
    b=np.array([.6,.9,1.2]);s=np.stack([np.ones(3),afi_ratio(b)],axis=-1)
    np.testing.assert_allclose(invert_afi(s*np.array([.1,100.,2000.])[:,None]),b,rtol=1e-13)
def test_exact_bloch_fixed_point():
    b=np.array([.6,1.,1.3]);t1=1000.;m0=2000.;a=b*np.pi/3;c=np.cos(a)
    e1=np.exp(-20/t1);e2=np.exp(-100/t1);z1=np.full(3,m0)
    for _ in range(1000):
        z2=m0+(z1*c-m0)*e1
        z1=m0+(z2*c-m0)*e2
    iterative=np.stack([z1*np.sin(a),z2*np.sin(a)],axis=-1)
    np.testing.assert_allclose(exact_steady_state(b,t1,m0),iterative,rtol=1e-13)
def test_short_tr_limit_and_bias():
    b=np.linspace(.5,1.4,21);exact=exact_steady_state(b,1000.,tr1_ms=.002,tr2_ms=.01)
    half=exact_steady_state(b,1000.,tr1_ms=.001,tr2_ms=.005)
    error=np.max(np.abs(exact[...,1]/exact[...,0]-afi_ratio(b)))
    half_error=np.max(np.abs(half[...,1]/half[...,0]-afi_ratio(b)))
    # AFI neglects finite-TR terms: test asymptotic convergence, not equality.
    assert 0<error<1e-6 and half_error<.51*error
    finite=exact_steady_state(b,1000.);bias=invert_afi(finite)-b
    assert np.max(np.abs(bias))>1e-4
def test_upstream_invalid_ratio_convention():
    assert invert_afi([1.,1.1])==0
    assert np.isnan(invert_afi([0.,0.]))
def test_metadata_and_observation_support():
    from src.preprocessing import load_data
    s,m=load_data(ROOT/'data/raw_data.npz');r=np.divide(s[...,1].astype(float),s[...,0],out=np.zeros(m.shape),where=s[...,0]!=0)
    expected=(s[...,0]>=.1*s[...,0].max())&(s[...,1]>=.1*s[...,1].max())&(r>.2)&(r<=(1+5*np.cos(np.pi/6))/(5+np.cos(np.pi/6)))
    np.testing.assert_array_equal(m,expected);assert m.sum()==80238
    text=(ROOT/'README.md').read_text(encoding='utf-8')
    assert '| 60 | degrees |' in text and '| 20 | ms |' in text and '| 100 | ms |' in text
def test_visualization_metric_formula():
    from src.visualization import metrics
    a=np.array([0.,1.,2.]);n,e=metrics(a,2*a)
    assert np.isclose(n,1.) and np.isclose(e,np.sqrt(5/3)/4)
