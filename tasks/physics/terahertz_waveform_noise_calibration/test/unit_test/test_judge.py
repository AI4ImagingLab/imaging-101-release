from pathlib import Path
import importlib.util,json,subprocess,sys,shutil,os
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('standalone_judge',ROOT/'evaluation/judge.py');judge=importlib.util.module_from_spec(spec);spec.loader.exec_module(judge)
def test_metric_known_cases():
    assert judge.relative_error(np.array([2.,4.]),np.array([1.,2.]))==1
    assert judge.relative_error(np.array([1j]),np.array([1j]))==0
    with pytest.raises(ValueError):judge.relative_error(np.zeros(2),np.zeros(2))
def invoke(prediction,report,cwd=None,script=None):
    return subprocess.run([sys.executable,str(script or ROOT/'evaluation/judge.py'),'--prediction',str(prediction),'--report',str(report)],cwd=cwd,capture_output=True,text=True,encoding='utf8',env=dict(os.environ,PYTHONIOENCODING='utf-8'))
def sample():
    with np.load(ROOT/'evaluation/baseline_reference.npz',allow_pickle=False) as f:return {k:f[k].copy() for k in f.files}
def test_pipeline_and_foreign_cwd(tmp_path):
    from src.main import run
    output=tmp_path/'fresh.npz';run(ROOT/'data/raw_data.npz',output)
    report=tmp_path/'nested/report.json';p=invoke(output,report,cwd=tmp_path);assert p.returncode==0,p.stderr
    data=json.loads(report.read_text());assert data['status']=='pass' and data['passed'];assert data['metrics']
@pytest.mark.parametrize('defect',['missing_file','missing_key','shape','dtype','nan','infinity','complex_dtype'])
def test_invalid_prediction(defect,tmp_path):
    a=sample();key=next(iter(a));path=tmp_path/'input.npz'
    if defect=='missing_key':a.pop(key)
    elif defect=='shape':a[key]=a[key].reshape(-1)
    elif defect=='dtype':a[key]=a[key].astype('complex64' if np.iscomplexobj(a[key]) else 'float32')
    elif defect=='complex_dtype':a[key]=a[key].astype('float64' if np.iscomplexobj(a[key]) else 'complex128')
    elif defect in ['nan','infinity']:a[key].flat[0]=np.nan if defect=='nan' else np.inf
    if defect!='missing_file':np.savez(path,**a)
    report=tmp_path/'report.json';p=invoke(path,report);assert p.returncode==2
    r=json.loads(report.read_text());assert r['status']=='error' and not r['passed'] and r['metrics']=={} and r['errors']
def test_every_required_quantity_matters(tmp_path):
    for key in sample():
        a=sample();a[key]=np.zeros_like(a[key]);path=tmp_path/'poor.npz';np.savez(path,**a);report=tmp_path/'poor.json';p=invoke(path,report)
        assert p.returncode==(2 if key in ('amplitude','noise') else 1),(key,p.stdout,p.stderr);assert not json.loads(report.read_text())['passed']
def test_missing_evaluator_assets_and_truth_priority(tmp_path):
    script=tmp_path/'judge.py';shutil.copy2(ROOT/'evaluation/judge.py',script);shutil.copy2(ROOT/'evaluation/metrics.json',tmp_path/'metrics.json')
    pred=tmp_path/'valid.npz';np.savez(pred,**sample());report=tmp_path/'report.json'
    assert invoke(pred,report,script=script).returncode==2
    shutil.copy2(ROOT/'evaluation/baseline_reference.npz',tmp_path/'baseline_reference.npz')
    np.savez(tmp_path/'ground_truth.npz',**{k:2*v for k,v in sample().items()})
    assert invoke(pred,report,script=script).returncode==2
def test_report_write_failure(tmp_path):
    occupied=tmp_path/'occupied';occupied.write_text('file')
    p=invoke(ROOT/'evaluation/baseline_reference.npz',occupied/'report.json');assert p.returncode==2 and 'Cannot write report' in p.stderr

def test_finite_huge_prediction_metric_overflow_is_reported(tmp_path):
    a=sample();key=next(iter(a));a[key]=np.full_like(a[key],1e300)
    pred=tmp_path/'huge.npz';np.savez(pred,**a);report=tmp_path/'report.json'
    p=invoke(pred,report);assert p.returncode==2
    result=json.loads(report.read_text(encoding='utf8'))
    assert result['status']=='error' and result['metrics']=={} and result['errors']
    assert 'variance' in result['errors'][0] or 'nonfinite' in result['errors'][0]
def test_main_import_preserves_backend():
    import matplotlib
    before=matplotlib.get_backend();import src.main
    assert matplotlib.get_backend()==before

@pytest.mark.parametrize('defect',['no_metrics','nonfinite_boundary','negative_boundary','missing_output_coverage'])
def test_undefined_metric_configuration_rejected(defect,tmp_path):
    script=tmp_path/'judge.py';shutil.copy2(ROOT/'evaluation/judge.py',script);config=json.loads((ROOT/'evaluation/metrics.json').read_text(encoding='utf8'))
    if defect=='no_metrics':config.pop('mean_nll_excess_boundary')
    elif defect=='nonfinite_boundary':config['mean_nll_excess_boundary']=float('inf')
    elif defect=='negative_boundary':config['mean_nll_excess_boundary']=-1
    else:config['output_schema']['unknown_output']=config['output_schema'].pop('noise')
    (tmp_path/'metrics.json').write_text(json.dumps(config),encoding='utf8');shutil.copy2(ROOT/'evaluation/baseline_reference.npz',tmp_path/'baseline_reference.npz');pred=tmp_path/'prediction.npz';np.savez(pred,**sample());report=tmp_path/'report.json';p=invoke(pred,report,script=script);assert p.returncode==2;data=json.loads(report.read_text());assert data['metrics']=={} and data['errors']


def test_known_gaussian_likelihood_better_noise_and_alias_gauge(tmp_path,monkeypatch):
    assets=tmp_path/'evaluation';assets.mkdir();(tmp_path/'data').mkdir()
    config=json.loads((ROOT/'evaluation/metrics.json').read_text());(assets/'metrics.json').write_text(json.dumps(config))
    shape=(50,267);observed=np.broadcast_to(np.where(np.arange(50)[:,None]%2,1.,-1.),shape).copy()
    np.savez(tmp_path/'data/raw_data.npz',waveforms=observed[None])
    correction=np.sqrt(50/49)
    def arrays(alpha):
        return dict(reconstruction=np.zeros((1,267)),amplitude=np.ones((1,50)),delay=np.zeros((1,50)),noise=np.array([[alpha,.01,.01]])*correction,expected_waveforms=np.zeros((1,50,267)),noise_std=np.full((1,50,267),alpha*correction))
    baseline=arrays(3.);np.savez(assets/'baseline_reference.npz',**baseline)
    monkeypatch.setattr(judge,'ROOT',assets)
    optimal=arrays(1.);optimal['delay'][0,1:]=.05003462406015038*267
    path=tmp_path/'optimal.npz';np.savez(path,**optimal)
    report=judge.evaluate(path);assert report['status']=='pass',report
    assert report['metrics']['mean_nll_excess']==pytest.approx(.5-(np.log(3)+1/18),abs=1e-12)
    poor=arrays(10.);np.savez(path,**poor);report=judge.evaluate(path)
    assert report['status']=='fail' and report['metrics']['mean_nll_excess']>1e-6
