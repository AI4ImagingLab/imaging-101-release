import numpy as np
from src.physics_model import spgr_signal,linearized_fit,REPETITION_TIME_MS,FLIP_ANGLES_DEG
def test_bloch_recurrence_independent():
    t1=1200.;m0=2000.;b1=.85
    e=np.exp(-15/t1);alpha=np.array([3.,20.])*np.pi/180*b1
    # Direct pulse/recovery fixed point, independently iterated to steady state.
    z=np.full(2,m0)
    for _ in range(5000):z=m0+(z*np.cos(alpha)-m0)*e
    np.testing.assert_allclose(spgr_signal(m0,t1,b1),z*np.sin(alpha),rtol=1e-12)
def test_analytic_inverse_and_units():
    t=np.array([300.,1000.,3000.]);m=np.array([500.,2000.,5000.]);b=np.array([.8,1.,1.2])
    signal=spgr_signal(m,t,b);T,M=linearized_fit(signal,b)
    np.testing.assert_allclose(T,t,rtol=1e-12);np.testing.assert_allclose(M,m,rtol=1e-12)
    assert np.all(signal>0)
def test_known_half_recovery():
    signal=spgr_signal(1000.,15/np.log(2),1.)
    a=np.array([3.,20.])*np.pi/180
    np.testing.assert_allclose(signal,1000*np.sin(a)*.5/(1-.5*np.cos(a)),rtol=1e-14)
def test_calibration_changes_lifetime():
    signal=spgr_signal(2000.,1000.,.8)
    corrected=linearized_fit(signal,.8)[0];wrong=linearized_fit(signal,1.)[0]
    assert np.isclose(corrected,1000) and abs(wrong-1000)>100
def test_metadata_mask_and_forward_consistency():
    from pathlib import Path
    from src.solvers import reconstruct
    root=Path(__file__).resolve().parents[2];r=reconstruct(root/'data/raw_data.npz')
    with np.load(root/'data/raw_data.npz') as f:
        m=f['tissue_mask'][0].astype(bool);s=f['measurements'][0];b=f['b1_map'][0]
        assert m.sum()==4668
    pred=spgr_signal(r['equilibrium_magnetization'][0][m],r['reconstruction'][0][m],b[m])
    np.testing.assert_allclose(pred,s[m],rtol=1e-10,atol=1e-8)
    for value in r.values():assert np.all(value[0][~m]==0)
    text=(root/'README.md').read_text(encoding='utf-8')
    assert '3, 20' in text and '| 15 | ms |' in text
def test_visualization_metrics():
    from src.visualization import metrics
    a=np.array([0.,1.,2.]);b=2*a;n,e=metrics(a,b)
    assert abs(n-1)<1e-14;assert abs(e-np.sqrt(5/3)/4)<1e-14
