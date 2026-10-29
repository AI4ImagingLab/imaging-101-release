from pathlib import Path
import importlib.util,json,subprocess,sys
import numpy as np
import pytest
from src.physics_model import solve_kepler,planet_velocity,transit_to_periastron
from src.solvers import INITIAL,negative_logposterior,submission,log_prior
from src.preprocessing import load_data
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('independent_judge',ROOT/'evaluation/judge.py');J=importlib.util.module_from_spec(spec);spec.loader.exec_module(J)
def reference():return J.load_arrays(ROOT/'evaluation/baseline_reference.npz')
def save(tmp_path,data):
    p=tmp_path/'prediction.npz';np.savez_compressed(p,**data);return p

def test_kepler_equation_and_velocity_units():
    m=np.linspace(0,2*np.pi,301);E=solve_kepler(m,.9)
    np.testing.assert_allclose(E-.9*np.sin(E),m,atol=1.1e-12,rtol=0)
    t=np.linspace(0,100,501);rv=planet_velocity(t,10,3,0,0,7)
    np.testing.assert_allclose(rv,7*np.cos(2*np.pi*(t-(3-2.5))/10),atol=1e-12)
    np.testing.assert_allclose(planet_velocity(t+10,10,3,.6,1.2,7),planet_velocity(t,10,3,.6,1.2,7),atol=1e-12)

def test_transit_velocity_and_orbit_average():
    P,tc,e,w,K=75.,120.,.7,1.3,3.
    assert abs(planet_velocity(np.array([tc]),P,tc,e,w,K)[0]-K*e*np.cos(w))<1e-12
    t=np.linspace(0,P,100001,endpoint=False)
    assert abs(planet_velocity(t,P,tc,e,w,K).mean())<1e-12

def test_judge_known_cases_and_periodic_conventions():
    r=reference();m=J.metrics(r,r);assert all(v==0 for v in m.values())
    p={k:v.copy() for k,v in r.items()};p['reconstruction'][0,:,1]+=r['reconstruction'][0,:,0];p['reconstruction'][0,:,3]+=2*np.pi
    assert J.metrics(p,r)['epoch_phase_abs_day']<1e-9
    assert J.metrics(p,r)['omega_phase_abs_rad']<1e-14
    p['instrument_offset']+=2;assert J.metrics(p,r)['offset_abs_mps']==2

@pytest.mark.parametrize('quantity',range(16))
def test_every_physical_output_is_scored(tmp_path,quantity):
    p=reference()
    if quantity<10:
        row,col=divmod(quantity,5);p['reconstruction'][0,row,col]+=[.01,.1,.001,.01,.01][col]
    elif quantity<13:p['instrument_offset'][0,quantity-10]+=.01
    else:p['instrument_jitter'][0,quantity-13]+=.01
    assert J.judge(save(tmp_path,p))['status']=='fail'

@pytest.mark.parametrize('kind',['missing_key','shape','float32','complex','nan','inf','object'])
def test_malformed_prediction(tmp_path,kind):
    p=reference()
    if kind=='missing_key':del p['instrument_jitter']
    elif kind=='shape':p['reconstruction']=p['reconstruction'][0]
    elif kind=='float32':p['reconstruction']=p['reconstruction'].astype(np.float32)
    elif kind=='complex':p['reconstruction']=p['reconstruction'].astype(complex)
    elif kind=='object':p['reconstruction']=p['reconstruction'].astype(object)
    else:p['reconstruction'][0,0,0]=np.nan if kind=='nan' else np.inf
    r=J.judge(save(tmp_path,p));assert r['status']=='error' and not r['metrics'] and r['errors']

def test_missing_prediction_evaluator_failure_and_bad_baseline(tmp_path):
    assert J.judge(tmp_path/'absent.npz')['status']=='error'
    assert J.judge(save(tmp_path,reference()),assets=tmp_path)['status']=='error'
    assert not J.judge(save(tmp_path,submission(INITIAL)))['passed']
    assert negative_logposterior(INITIAL,load_data())>1045
    p=INITIAL.copy();p[2]=1;p[3]=1;assert not np.isfinite(log_prior(p))

def test_cli_exit_codes_foreign_cwd_and_report_failure(tmp_path):
    path=save(tmp_path,reference());report=tmp_path/'nested/report.json'
    cmd=[sys.executable,str(ROOT/'evaluation/judge.py'),'--prediction',str(path),'--report',str(report)]
    assert subprocess.run(cmd,cwd=tmp_path,capture_output=True).returncode==0
    path=save(tmp_path,submission(INITIAL));assert subprocess.run(cmd,cwd=tmp_path,capture_output=True).returncode==1
    path.unlink();assert subprocess.run(cmd,cwd=tmp_path,capture_output=True).returncode==2
    save(tmp_path,reference());cmd[-1]=str(path/'bad/report.json');assert subprocess.run(cmd,cwd=tmp_path,capture_output=True).returncode==2


def test_visual_metrics_and_phase_summary():
    from src.visualization import parameter_errors,residual_diagnostics,phase_bins
    with np.load(ROOT/'test/fixtures/refined_map_oracle.npz',allow_pickle=False) as z:v=z['vary']
    r,s,w=residual_diagnostics(load_data(),v)
    assert abs(w-J.observation_metrics(reference(),np.load(ROOT/'data/raw_data.npz',allow_pickle=False))['weighted_residual_rms'])<1e-12
    assert not np.any(parameter_errors(v,v))
    p=v.copy();p[10]+=2;assert parameter_errors(p,v)[10]==2
    phase=np.linspace(.01,.99,100);bc,by,be=phase_bins(phase,np.ones(100)*3)
    assert len(bc)>10 and np.all((bc>=-.5)&(bc<=.5))
    np.testing.assert_allclose(by,3,atol=1e-14);np.testing.assert_allclose(be,0,atol=1e-14)

def test_all_original_fixture_consumers_and_public_metadata():
    d=load_data();text=(ROOT/'README.md').read_text(encoding='utf8')
    with np.load(ROOT/'test/fixtures/initial_oracle.npz',allow_pickle=False) as z:
        np.testing.assert_array_equal(INITIAL,z['initial'])
        for v,l in zip(z['probe_vectors'],z['probe_loss']):assert abs(negative_logposterior(v,d)-l)<1e-8
        from src.physics_model import components
        np.testing.assert_allclose(components(d['time'],INITIAL).sum(0),z['model'],atol=1e-10)
    for k,a in d.items():
        assert f'| {k} | (1, 401) | {a.dtype} |' in text
    assert '2450275.9700771; 2457292.6796628' in text
    assert np.bincount(d['instrument']).tolist()==[52,276,73]
    assert '52 observations' in text and '276 observations' in text and '73 observations' in text
    assert '0.798407554626 to 2.96615958214' in text

def test_bad_npz_and_undefined_metric_configuration(tmp_path):
    p=tmp_path/'bad.npz';p.write_text('not npz');r=J.judge(p);assert r['status']=='error' and not r['metrics']
    p=save(tmp_path,reference())
    np.savez_compressed(tmp_path/'baseline_reference.npz',**reference())
    (tmp_path/'metrics.json').write_text(json.dumps({'thresholds':{'x':-1}}))
    assert J.judge(p,assets=tmp_path)['status']=='error'


def test_likelihood_normalisation_at_zero_residual():
    from src.physics_model import forward
    v=INITIAL.copy();d=load_data();d['velocity']=forward(d['time'],d['instrument'],v)
    var=d['uncertainty']**2+v[13:16][d['instrument']]**2
    expected=.5*np.log(2*np.pi*var).sum()-log_prior(v)
    assert abs(negative_logposterior(v,d)-expected)<1e-10

def test_explicit_ground_truth_priority_contract_fixture(tmp_path):
    # A controlled evaluator-priority case, not physical truth for this benchmark.
    import shutil
    assets=tmp_path/'case/evaluation';assets.mkdir(parents=True)
    data=assets.parent/'data';data.mkdir()
    shutil.copy2(ROOT/'data/raw_data.npz',data/'raw_data.npz')
    shutil.copy2(ROOT/'evaluation/metrics.json',assets/'metrics.json')
    ref=reference();np.savez_compressed(assets/'baseline_reference.npz',**ref)
    altered={k:a.copy() for k,a in ref.items()};altered['reconstruction'][0,0,4]+=.1
    np.savez_compressed(assets/'ground_truth.npz',**altered)
    report=J.judge(save(tmp_path,ref),assets=assets)
    assert report['status']=='fail' and abs(report['metrics']['amplitude_abs_mps']-.1)<1e-12
