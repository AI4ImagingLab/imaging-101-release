from pathlib import Path
import numpy as np
from scipy.integrate import quad
from src.physics_model import buxton_signal, LABEL_DURATION, POST_LABEL_DELAY
from src.preprocessing import load_observations
from src.solvers import reconstruct
ROOT=Path(__file__).resolve().parents[2]

def test_upstream_forward_stages():
    with np.load(ROOT/'test/fixtures/upstream_forward.npz',allow_pickle=False) as f:
        np.testing.assert_array_equal(f['input_ld'],LABEL_DURATION)
        np.testing.assert_array_equal(f['input_pld'],POST_LABEL_DELAY)
        for flow,att,expected in zip(f['param_flow'],f['param_att'],f['output_signal']):
            np.testing.assert_allclose(buxton_signal(LABEL_DURATION,POST_LABEL_DELAY,float(f['param_m0']),float(flow),float(att)),expected,rtol=3e-14,atol=1e-8)

def test_independent_bolus_integral_and_units():
    for flow in [1e-6,1e-5,3e-5]:
        eff=1/(1/1650+flow/0.98)
        for att in [100.,600.,1000.,2000.,4000.]:
            expected=[]
            for tau,w in zip(LABEL_DURATION,POST_LABEL_DELAY):
                t=tau+w
                integral=0 if t<=att else quad(lambda s:np.exp(-(t-s)/eff),att,min(t,att+tau),epsabs=1e-10)[0]
                expected.append(2*.85*8e7*flow*np.exp(-att/1650)*integral)
            np.testing.assert_allclose(buxton_signal(LABEL_DURATION,POST_LABEL_DELAY,8e7,flow,att),expected,rtol=1e-10,atol=1e-8)
    assert 1e-5*100*60000==60

def test_full_pipeline_upstream_parity_and_input_identity():
    data=load_observations(ROOT/'data/raw_data.npz')
    result=reconstruct(**data)
    with np.load(ROOT/'test/fixtures/upstream_pipeline.npz',allow_pickle=False) as f:
        np.testing.assert_array_equal(f['reconstruction'],f['raw_flow'].astype(np.float64)*6e6)
        for key in ['reconstruction','arterial_transit_time']:
            ref=f[key]
            np.testing.assert_allclose(result[key],ref,rtol=3e-6,atol=1e-5)
            assert np.sqrt(np.mean((result[key]-ref)**2))/np.ptp(ref)<1e-6
    mask=data['mask'][0]>0
    assert np.all(result['reconstruction'][0][~mask]==0)
    assert np.all(result['arterial_transit_time'][0][~mask]==0)
