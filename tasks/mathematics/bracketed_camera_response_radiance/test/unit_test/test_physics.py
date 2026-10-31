import numpy as np
import pytest
from src.physics_model import response_system,triangular_weights,predict_codes
from src.preprocessing import sample_stack
from src.solvers import fit_response,merge_exposures

def test_weight_endpoints_midpoint():
    np.testing.assert_array_equal(triangular_weights(np.array([0,.25,.5,.75,1.])),[0,.5,1,.5,0])

def test_second_difference_and_gauge():
    samples=np.array([[20.,80.,140.],[50.,110.,170.]])
    a,b=response_system(samples,np.exp([0,1,2]),smoothing=3.)
    assert a.shape==(263,258) and a[6,128]==1 and b[6]==0
    np.testing.assert_array_equal(a[-2:],0)
    q=np.r_[np.arange(256),[0.,0.]]
    np.testing.assert_allclose((a@q)[7:-2],0,atol=1e-12)
    np.testing.assert_allclose(a[0,[20,256]],[40/255,-40/255])

def test_independent_linear_log_response_recovers_exact_data():
    times=np.exp(np.array([-1.,0.,1.]));z=np.arange(30,180,10)[:,None]+np.array([-20,0,20])[None]
    g=fit_response(np.repeat(z[...,None],3,axis=2),times)
    np.testing.assert_allclose(g,(np.arange(256)[:,None]-128)/20*np.ones((1,3)),atol=1e-9)

def test_sampling_any_channel_monotonic_and_column_major():
    f=np.zeros((3,3,4,3),dtype=np.uint8)
    for j in range(3):f[j,...,0]=30-10*j
    f[:,:,1,0]=0
    s=sample_stack(f,np.array([3.,2.,1.]),256)
    assert s.shape==(9,3,3)
    np.testing.assert_array_equal(s[:,:,0],np.tile([30,20,10],(9,1)))

def test_photometric_reciprocity_and_clipping():
    response=np.exp((np.arange(256)[:,None]-128)/20)*np.ones((1,3));e=np.ones((2,2,3))
    predicted=predict_codes(e,response,np.exp([-1,0,1]))
    np.testing.assert_allclose(predicted[:,0,0,0],[108,128,148],atol=1e-12)
    np.testing.assert_array_equal(predict_codes(e*1e30,response,np.array([1.])),255)

@pytest.mark.parametrize('kind',['saturated','dark'])
def test_merge_endpoint_fallback(kind):
    frames=np.full((3,2,2,3),255 if kind=='saturated' else 0,dtype=np.uint8)
    response=np.exp((np.arange(256)[:,None]-128)/20)*np.ones((1,3))
    result=merge_exposures(frames,np.array([4.,2.,1.]),response)
    assert np.all(np.isfinite(result)) and np.all(result>0)

def test_fit_stationarity_on_bound_frozen_samples():
    from pathlib import Path
    with np.load(Path(__file__).resolve().parents[1]/'fixtures/upstream_stages.npz',allow_pickle=False) as f:
        a,b=response_system(f['output_samples'][...,0],f['input_exposure_times'])
    from scipy.linalg import lstsq
    q=lstsq(a,b)[0]
    assert np.linalg.norm(a.T@(a@q-b),ord=np.inf)<1e-7

@pytest.mark.parametrize('bad',[-1,256,3.5])
def test_invalid_sample_code_rejected(bad):
    with pytest.raises(ValueError):response_system(np.array([[bad,5]]),np.array([1.,2.]))

def test_visualization_metrics_known_case_and_log_consistency():
    from src.visualization import agreement
    from src.physics_model import log_exposure_residual
    a=np.array([1.,2.,3.]);m=agreement(2*a,a)
    assert m['ncc']==pytest.approx(1) and m['log_rmse']==pytest.approx(np.log(2))
    response=np.exp((np.arange(256)[:,None]-128)/20)*np.ones((1,3))
    frames=np.full((3,2,2,3),128,dtype=np.uint8)
    assert log_exposure_residual(frames,np.ones(3),np.ones((2,2,3)),response)==0

def test_public_exposure_constants_agree():
    from pathlib import Path
    from src.preprocessing import EXPOSURE_TIMES,load_observations
    root=Path(__file__).resolve().parents[2]
    _,times=load_observations(root/'data/raw_data.npz')
    np.testing.assert_array_equal(times,EXPOSURE_TIMES)
    text=(root/'README.md').read_text(encoding='utf8')
    row=next(x for x in text.splitlines() if x.startswith('| Exposure order |'))
    np.testing.assert_array_equal(np.array([float(x) for x in row.split('|')[2].split(',')]),times)
