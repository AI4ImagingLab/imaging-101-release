from pathlib import Path
import numpy as np
from src.preprocessing import load_data
from src.physics_model import solve_kepler,components
from src.solvers import negative_logposterior,log_prior,reconstruct,reconstruct_native
F=Path(__file__).resolve().parents[1]/'fixtures'
def test_frozen_upstream_stages():
    d=load_data()
    with np.load(F/'stages_oracle.npz',allow_pickle=False) as z:
        for e,E in zip([0,.1,.7,.99],z['eccentric_anomaly']):np.testing.assert_allclose(solve_kepler(z['mean_anomaly'],e),E,atol=1e-11,rtol=0)
        for v,c,l,p in zip(z['probes'],z['components'],z['negative_logposterior'],z['prior_terms']):
            np.testing.assert_allclose(components(d['time'],v),c,atol=1e-10,rtol=0)
            assert abs(negative_logposterior(v,d)-l)<1e-8
            assert abs(log_prior(v)-p.sum())<1e-10

def test_native_early_stop_baseline_against_independent_original():
    d=load_data();v,r=reconstruct_native(d)
    with np.load(F/'map_oracle.npz',allow_pickle=False) as z:
        assert np.max(np.abs(v-z['vary']))<.001
        assert np.sqrt(np.mean((components(d['time'],v).sum(0)-z['model'])**2))<.001
        # CompositeLikelihood concatenates observations in native k,j,a order.
        order=np.r_[np.where(d['instrument']==0)[0],np.where(d['instrument']==1)[0],np.where(d['instrument']==2)[0]]
        from src.physics_model import forward
        np.testing.assert_allclose((d['velocity']-forward(d['time'],d['instrument'],v))[order],z['residual'],atol=.001,rtol=0)
        sigma=np.sqrt(d['uncertainty']**2+v[13:16][d['instrument']]**2)
        np.testing.assert_allclose(sigma[order],z['errorbars'],atol=.001,rtol=0)
        expected=np.r_[v[0],v[5],v[1],v[6],0.,0.,v[10:16],v[2:5],v[7:10]]
        np.testing.assert_allclose(expected,z['all_values'],atol=.001,rtol=0)
    assert r['success']

def test_public_native_observation_binding():
    d=load_data()
    with np.load(F/'initial_oracle.npz',allow_pickle=False) as z:
        for k,j in [('time','time'),('velocity','velocity'),('uncertainty','error'),('instrument','instrument')]:np.testing.assert_array_equal(d[k],z[j])


def test_full_precision_pipeline_against_independent_original_multistart():
    d=load_data();v,r=reconstruct(d)
    with np.load(F/'refined_map_oracle.npz',allow_pickle=False) as z:
        np.testing.assert_allclose(v,z['vary'],atol=.001,rtol=0)
        assert np.sqrt(np.mean((components(d['time'],v).sum(0)-z['model'])**2))<.001
        order=np.concatenate([np.where(d['instrument']==k)[0] for k in range(3)])
        from src.physics_model import forward
        np.testing.assert_allclose((d['velocity']-forward(d['time'],d['instrument'],v))[order],z['residual'],atol=.001,rtol=0)
        sigma=np.sqrt(d['uncertainty']**2+v[13:16][d['instrument']]**2)
        np.testing.assert_allclose(sigma[order],z['errorbars'],atol=.001,rtol=0)
        expected=np.r_[v[0],v[5],v[1],v[6],0.,0.,v[10:16],v[2:5],v[7:10]]
        np.testing.assert_allclose(expected,z['all_values'],atol=.001,rtol=0)
    assert r['success'] and len(r['runs'])==4
    assert max(abs(q['negative_logposterior']-r['negative_logposterior']) for q in r['runs'])<1e-7
    with np.load(F/'refined_multistart.npz',allow_pickle=False) as z:
        assert z['starts'].shape==(4,16)
        assert np.ptp(z['solutions'],axis=0).max()<.001
        for row in z['solutions']:assert abs(negative_logposterior(row,d)-r['negative_logposterior'])<1e-7
        assert np.array_equal(z['centre'],__import__('src.solvers',fromlist=['INITIAL']).INITIAL)
        assert np.array_equal(z['scale'],__import__('src.solvers',fromlist=['COORDINATE_SCALE']).COORDINATE_SCALE)
