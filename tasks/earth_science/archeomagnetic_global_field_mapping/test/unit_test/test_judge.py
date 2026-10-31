import importlib.util,json,subprocess,sys
from pathlib import Path
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('judge',ROOT/'evaluation/judge.py'); judge=importlib.util.module_from_spec(spec); spec.loader.exec_module(judge)

def test_metrics_known_case():
    out=judge.metrics([0.,2.],[0.,1.]); assert out['ncc']==pytest.approx(1.); assert out['nrmse']==pytest.approx(np.sqrt(.5))
    assert judge.metrics([0.,0.],[0.,1.])['ncc']==0

def test_valid_reference_and_bad_quality(tmp_path):
    ref=np.load(ROOT/'evaluation/baseline_reference.npz'); good=tmp_path/'good.npz'; np.savez(good,**{k:ref[k] for k in ref.files})
    assert judge.evaluate(good)[1]==0
    bad=tmp_path/'bad.npz'; np.savez(bad,**{k:np.zeros_like(ref[k]) for k in ref.files})
    report,code=judge.evaluate(bad); assert code==1 and report['status']=='fail' and report['metrics']

@pytest.mark.parametrize('error',['missing','key','shape','dtype','complex','nan','inf'])
def test_malformed(error,tmp_path):
    ref=np.load(ROOT/'evaluation/baseline_reference.npz'); payload={k:ref[k].copy() for k in ref.files}; path=tmp_path/'bad.npz'
    if error=='missing': pass
    else:
        if error=='key': payload.pop('coefficients')
        elif error=='shape': payload['reconstruction']=payload['reconstruction'][0]
        elif error=='dtype': payload['reconstruction']=payload['reconstruction'].astype('f4')
        elif error=='complex': payload['reconstruction']=payload['reconstruction'].astype('c16')
        elif error=='nan': payload['reconstruction'].flat[0]=np.nan
        elif error=='inf': payload['reconstruction'].flat[0]=np.inf
        np.savez(path,**payload)
    report,code=judge.evaluate(path); assert code==2 and report['metrics']=={} and report['errors']

def test_missing_evaluator_assets(monkeypatch,tmp_path):
    monkeypatch.setattr(judge,'HERE',tmp_path)
    report,code=judge.evaluate(ROOT/'output/reconstruction.npz'); assert code==2 and report['metrics']=={}

def test_report_write_failure(tmp_path):
    directory=tmp_path/'report'; directory.mkdir()
    assert judge.main(['--prediction',str(ROOT/'evaluation/baseline_reference.npz'),'--report',str(directory)])==2

def test_visualization_metric_consistency():
    from src.visualization import metric_pair
    for a,b in [([0.,2.],[0.,1.]),([1.,3.,2.],[2.,1.,4.])]:
        assert metric_pair(a,b)==pytest.approx(judge.metrics(a,b))

def test_observation_metrics_independent_source_values():
    reference=np.load(ROOT/'evaluation/baseline_reference.npz')
    result=judge.observation_metrics(reference['coefficients'])
    assert result==pytest.approx({'intensity_whitened_rms':1.6060121973684938,
        'inclination_whitened_rms':1.70794816253306,
        'declination_whitened_rms':1.2452833003510602},rel=1e-12,abs=1e-12)

def test_observation_metrics_analytic_equatorial_dipole(tmp_path):
    path=tmp_path/'public.npz'
    np.savez(path,intensity=np.array([[[0.,0.,0.,30000.,1.]]]),
        inclination=np.array([[[0.,0.,0.,0.,1.]]]),
        declination=np.array([[[0.,0.,0.,360.,1.]]]))
    coefficients=np.zeros((1,43,15))
    # The public ellipsoid equator has radius6378.137km, so correct for radial decay.
    coefficients[:,:,0]=-30000*(6378.137/6371.2)**3
    result=judge.observation_metrics(coefficients,path)
    assert max(result.values())<1e-9

def test_observation_boundaries_actually_control_pass(monkeypatch,tmp_path):
    import shutil
    shutil.copyfile(ROOT/'evaluation/baseline_reference.npz',tmp_path/'baseline_reference.npz')
    config=json.loads((ROOT/'evaluation/metrics.json').read_text(encoding='utf8'))
    config['intensity_whitened_rms_boundary']=1.
    (tmp_path/'metrics.json').write_text(json.dumps(config),encoding='utf8')
    monkeypatch.setattr(judge,'HERE',tmp_path)
    report,code=judge.evaluate(ROOT/'evaluation/baseline_reference.npz')
    assert code==1 and report['status']=='fail'
    assert report['metrics']['intensity_whitened_rms']>report['thresholds']['intensity_whitened_rms']['boundary']

def test_bad_coefficients_increase_public_observation_residual():
    result=judge.observation_metrics(np.zeros((1,43,15)))
    config=json.loads((ROOT/'evaluation/metrics.json').read_text(encoding='utf8'))
    assert all(result[k]>config[k+'_boundary'] for k in result)

@pytest.mark.parametrize('problem',['nan','inf','missing','negative','ncc_range','negative_ncc_range','observation_negative','string','boolean'])
def test_invalid_metric_configuration(problem,monkeypatch,tmp_path):
    import shutil
    shutil.copyfile(ROOT/'evaluation/baseline_reference.npz',tmp_path/'baseline_reference.npz')
    config=json.loads((ROOT/'evaluation/metrics.json').read_text(encoding='utf8'))
    if problem=='nan': config['relative_energy_excess_boundary']=float('nan')
    elif problem=='inf': config['intensity_whitened_rms_boundary']=float('inf')
    elif problem=='missing': config.pop('declination_whitened_rms_boundary')
    elif problem=='negative': config['relative_energy_excess_boundary']=-1.
    elif problem=='ncc_range': config['map_consistency_nT_boundary']=-1
    elif problem=='negative_ncc_range': config['map_consistency_nT_boundary']=float('nan')
    elif problem=='observation_negative': config['inclination_whitened_rms_boundary']=-1.
    elif problem=='string': config['relative_energy_excess_boundary']='small'
    else: config['relative_energy_excess_boundary']=True
    (tmp_path/'metrics.json').write_text(json.dumps(config),encoding='utf8')
    monkeypatch.setattr(judge,'HERE',tmp_path)
    report,code=judge.evaluate(ROOT/'evaluation/baseline_reference.npz')
    assert code==2 and report['status']=='error' and report['metrics']=={} and report['errors']

def test_nonfinite_computed_metric_returns_evaluator_error(tmp_path):
    reference=np.load(ROOT/'evaluation/baseline_reference.npz')
    prediction={key:reference[key].copy() for key in reference.files}
    prediction['reconstruction'].fill(1e308)
    path=tmp_path/'overflow.npz'; np.savez(path,**prediction)
    with np.errstate(over='ignore',invalid='ignore'):
        report,code=judge.evaluate(path)
    assert code==1 and report['metrics']['map_consistency_nT']>1e300


def test_prior_exact_constant_gauss_modes():
    c=np.zeros((1,43,15));c[:,:,0]=-30000
    assert judge.prior_energy(c)==pytest.approx(0,abs=1e-20)
    c[:]=0;c[:,:,3]=1
    q=6371.2/3485.;expected=1e-13*4000*4*np.pi*q**7*3*5*7/2
    assert judge.prior_energy(c)==pytest.approx(expected,rel=1e-12)


def test_reference_scale_optimization_and_map_consistency(tmp_path):
    from scipy.optimize import minimize_scalar
    with np.load(ROOT/'evaluation/baseline_reference.npz',allow_pickle=False) as z:baseline={k:z[k] for k in z.files}
    original=judge.energy(baseline['coefficients'])
    fit=minimize_scalar(lambda scale:judge.energy(baseline['coefficients']*scale),bounds=(.99,1.01),method='bounded',options={'xatol':1e-12})
    assert fit.fun<=original+1e-8
    coefficients=baseline['coefficients']*fit.x
    year,lat,lon=np.meshgrid([-1500.,-500.,500.,1000.,1700.],np.linspace(-87.5,87.5,36),np.arange(-177.5,180.,5.),indexing='ij')
    xyz=judge.field_at(coefficients,year.ravel(),lat.ravel(),lon.ravel())
    field=np.stack([v.reshape(5,36,72) for v in xyz],axis=1)[None]
    path=tmp_path/'different.npz';np.savez(path,reconstruction=field,coefficients=coefficients)
    report,code=judge.evaluate(path);assert code==0,report
    field[0,0,0,0,0]+=.01
    np.savez(path,reconstruction=field,coefficients=coefficients)
    report,code=judge.evaluate(path);assert code==1 and report['metrics']['map_consistency_nT']>.001
