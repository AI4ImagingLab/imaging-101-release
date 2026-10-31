from pathlib import Path
import numpy as np
from src.physics_model import eccentric_anomaly,periastron,T0,orbit,photometry
from src.preprocessing import load_data
from src.solvers import models,residuals
ROOT=Path(__file__).resolve().parents[2]

def test_original_orbital_helpers():
    with np.load(ROOT/'test/fixtures/v2_original_orbital_helpers.npz') as f:
        assert np.max(abs(eccentric_anomaly(f['input_mean_anomaly'],f['input_eccentricity'])-f['output_eccentric_anomaly']))<2e-10
        got=[periastron(T0,p[3]**2+p[4]**2,np.arctan2(p[3],p[4]),np.deg2rad(p[2])) for p in f['param_points']]
        assert np.max(abs(got-f['output_periastron_epoch']))<5e-8

def test_full_six_group_forward_and_likelihood():
    groups=load_data(ROOT/'data/raw_data.npz')
    with np.load(ROOT/'test/fixtures/v2_original_forward_likelihood.npz') as f:
        for k,p in enumerate(f['param_points']):
            for gi,(name,pred) in enumerate(zip(['wasp','nites','oed','byu','kpno','rv'],models(p,groups))):
                np.testing.assert_allclose(pred,f['output_model_'+name][k],rtol=0,atol=1e-5 if gi==5 else 1e-7)
            chi,z,pred=residuals(p,groups,True)
            np.testing.assert_allclose(chi,f['output_chi_squared'][k],rtol=0,atol=.02)
            np.testing.assert_allclose(z,f['output_offsets'][k],rtol=0,atol=1e-7)
            assert abs(-.5*chi.sum()-f['output_log_likelihood'][k])<.06

def test_original_precision_and_schedules():
    groups=load_data(ROOT/'data/raw_data.npz')
    with np.load(ROOT/'test/fixtures/v2_original_forward_likelihood.npz') as f:
        for (t,y,s,wt,flag),name in zip(groups,['wasp','nites','oed','byu','kpno','rv']):
            assert np.array_equal(t,f['input_times_'+name])
            assert np.array_equal(y,f['input_values_'+name])
            assert np.array_equal(s,f['input_errors_'+name])
        for gi,name in [(1,'nites'),(4,'kpno')]:assert np.array_equal(groups[gi][4]==1,f['param_n_int_'+name]>0)
