import importlib.util,json,subprocess,sys
from pathlib import Path
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('independent_task_judge',ROOT/'evaluation/judge.py')
judge=importlib.util.module_from_spec(spec);spec.loader.exec_module(judge)
def reference():
    with np.load(ROOT/'evaluation/baseline_reference.npz',allow_pickle=False) as f:return {k:f[k].copy() for k in f.files}
def save(tmp_path,values):
    p=tmp_path/'prediction.npz';np.savez_compressed(p,**values);return p
def test_independent_known_metrics():
    a=np.array([1.,2.,3.]);ncc,nrmse=judge.metric_pair(a,a)
    assert ncc==pytest.approx(1);assert nrmse==0
    assert judge.metric_pair(np.zeros(3),a)==(0.,pytest.approx(np.sqrt(14/3)/2))
    assert judge.metric_pair(a*2,a)[1]==pytest.approx(np.sqrt(14/3)/2)
    with pytest.raises(ValueError):judge.metric_pair(a,np.ones(3))
def test_reference_pass_and_zero_fails(tmp_path):
    a=reference();report,code=judge.evaluate(save(tmp_path,a));assert code==0 and report['passed']
    report,code=judge.evaluate(save(tmp_path,{k:np.zeros_like(v) for k,v in a.items()}))
    assert code==1 and report['status']=='fail' and report['metrics']
@pytest.mark.parametrize('case',['missing','key','shape','dtype','nan','inf','complex','object'])
def test_malformed_inputs(tmp_path,case):
    a=reference();key=next(iter(a))
    if case=='missing':p=tmp_path/'not_here.npz'
    else:
        if case=='key':a.pop(key)
        elif case=='shape':a[key]=a[key][...,0]
        elif case=='dtype':a[key]=np.zeros(a[key].shape,dtype='int32')
        elif case in ['nan','inf']:
            a[key]=a[key].astype('float64');a[key].flat[0]=float(case)
        elif case=='complex':a[key]=a[key].astype('complex128')
        elif case=='object':a[key]=a[key].astype(object)
        p=save(tmp_path,a)
    report,code=judge.evaluate(p);assert code==2 and report['status']=='error'
    assert not report['metrics'] and report['errors']
def test_truth_priority_and_evaluator_error(tmp_path):
    a=reference();prediction=save(tmp_path,a);assets=tmp_path/'assets';assets.mkdir();(tmp_path/'data').mkdir(exist_ok=True);(tmp_path/'data/raw_data.npz').write_bytes((ROOT/'data/raw_data.npz').read_bytes())
    (assets/'metrics.json').write_bytes((ROOT/'evaluation/metrics.json').read_bytes())
    np.savez_compressed(assets/'baseline_reference.npz',**a)
    truth={k:(v.astype(float)*.5).astype(v.dtype) for k,v in a.items()}
    np.savez_compressed(assets/'ground_truth.npz',**truth)
    assert judge.evaluate(prediction,assets)[1]==1
    (assets/'ground_truth.npz').unlink();(assets/'baseline_reference.npz').unlink()
    assert judge.evaluate(prediction,assets)[1]==2
def test_cli_paths_independent_of_current_directory(tmp_path):
    prediction=save(tmp_path,reference());report=tmp_path/'new'/'report.json'
    r=subprocess.run([sys.executable,str(ROOT/'evaluation/judge.py'),'--prediction',str(prediction),'--report',str(report)],cwd=tmp_path,capture_output=True,text=True)
    assert r.returncode==0,r.stderr
    assert json.loads(report.read_text(encoding='utf8'))['status']=='pass'
def test_cli_report_write_failure(tmp_path):
    prediction=save(tmp_path,reference());report=tmp_path/'directory';report.mkdir()
    r=subprocess.run([sys.executable,str(ROOT/'evaluation/judge.py'),'--prediction',str(prediction),'--report',str(report)],capture_output=True,text=True)
    assert r.returncode==2 and 'Cannot write report' in r.stderr

def test_visualization_and_judge_share_known_metric_convention():
    import sys
    sys.path.insert(0,str(ROOT))
    from src.visualization import metrics
    a=np.array([1.,2.,3.]);b=np.array([2.,4.,6.])
    assert metrics(a,b)==pytest.approx(judge.metric_pair(a,b))

def test_invalid_metric_boundaries_are_evaluator_errors(tmp_path):
    a=reference();prediction=save(tmp_path,a);assets=tmp_path/'assets';assets.mkdir();(tmp_path/'data').mkdir(exist_ok=True);(tmp_path/'data/raw_data.npz').write_bytes((ROOT/'data/raw_data.npz').read_bytes())
    np.savez_compressed(assets/'baseline_reference.npz',**a)
    config=json.loads((ROOT/'evaluation/metrics.json').read_text(encoding='utf8'))
    config['nrmse_boundary']=-1
    (assets/'metrics.json').write_text(json.dumps(config),encoding='utf8')
    report,code=judge.evaluate(prediction,assets)
    assert code==2 and not report['metrics']

def test_observation_consistency_from_independent_reference():
    config=json.loads((ROOT/'evaluation/metrics.json').read_text(encoding='utf8'))
    measured=judge.observation_metric(reference(),ROOT/'data/raw_data.npz')
    assert measured==pytest.approx(config['observation_consistency']['original_measurement'],rel=1e-12,abs=1e-15)
