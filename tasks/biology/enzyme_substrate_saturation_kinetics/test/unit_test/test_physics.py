from pathlib import Path
import numpy as np
from src.physics_model import rate,jacobian,profile
from src.solvers import ecb,intersections,line_fit
from src.preprocessing import load_data
ROOT=Path(__file__).resolve().parents[2]
def test_forward_limits_and_half_saturation():
    assert rate(0,[2,12])==0
    assert rate(2,[2,12])==6
    assert abs(rate(1e12,[2,12])-12)<1e-9
def test_analytic_jacobian_and_loss_sign():
    s=np.array([.05,1.,3.,20.]);p=np.array([2.,12.]);obs=rate(s,p)+np.array([1,-2,3,-4])
    j=jacobian(s,p)
    for k in range(2):
        delta=np.eye(2)[k]*1e-5
        np.testing.assert_allclose(j[:,k],(rate(s,p+delta)-rate(s,p-delta))/2e-5,rtol=1e-8,atol=1e-9)
        gradient=2*j.T@(rate(s,p)-obs)
        finite=(sum((rate(s,p+delta)-obs)**2)-sum((rate(s,p-delta)-obs)**2))/2e-5
        np.testing.assert_allclose(gradient[k],finite,rtol=1e-8)
def test_ecb_original_bug_is_explicit():
    s=np.array([2.,4.]);v=np.array([6.,8.])
    np.testing.assert_array_equal(ecb(s,v),[2,14])
    np.testing.assert_array_equal(ecb(s,v,corrected=True),[2,12])
    np.testing.assert_allclose(intersections(s,v,True),intersections(s[::-1],v[::-1],True))
    assert not np.allclose(intersections(s,v),intersections(s[::-1],v[::-1]))
def test_profile_linear_elimination():
    s,v=load_data(ROOT/'data/raw_data.npz');k=np.array([1.,2.,8.])
    vmax,rss=profile(s,v[0],k)
    for i in range(3):
        a=s/(k[i]+s)
        assert abs(a@(v[0]-vmax[i]*a))<1e-10
        assert rss[i]<=sum((v[0]-(vmax[i]+1)*a)**2)
def test_weighted_line_closed_case():
    x=np.arange(1.,6.);np.testing.assert_allclose(line_fit(x,2+3*x,x**4),[2,3],atol=1e-12)
def test_raw_contract_and_paper_table2():
    s,v=load_data(ROOT/'data/raw_data.npz')
    np.testing.assert_array_equal(s,[.05,.1,.25,.5,1,2.5,5,8,20,30])
    np.testing.assert_allclose(v[5],[3,6,17,31,48,101,121,139,152,181],rtol=0,atol=1e-12)
    assert v[1,7]==12.9
    readme=(ROOT/'README.md').read_text(encoding='utf8')
    for key in ['substrate','rates','maximum_rate','fitted_rate','micromolar/min','30','0.05'] :assert key in readme
