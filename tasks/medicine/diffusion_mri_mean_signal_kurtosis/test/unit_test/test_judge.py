from pathlib import Path
import importlib.util,json,subprocess,sys,shutil
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('independent_judge',ROOT/'evaluation/judge.py');judge=importlib.util.module_from_spec(spec);spec.loader.exec_module(judge)
def arrays():
    with np.load(ROOT/'evaluation/baseline_reference.npz',allow_pickle=False) as d:return {k:d[k].copy() for k in d.files}

def test_known_metrics():
    assert judge.score(np.array([2.,0]),np.array([1.,0]))==1
    assert judge.angular_score(np.array([.2]),np.array([.2+np.pi]),np.array([.1]))<1e-14

@pytest.mark.parametrize('kind',['missing','key','shape','dtype','complex','nan','inf','poor','good'])
def test_prediction_contract(tmp_path,kind):
    a=arrays();key='reconstruction';target=tmp_path/'prediction.npz'
    if kind=='key':a['other']=a.pop(key)
    if kind=='shape':a[key]=a[key].ravel()
    if kind=='dtype':a[key]=a[key].astype('float32' if a[key].dtype==np.float64 else 'float64')
    if kind=='complex':a[key]=a[key].astype(complex)
    if kind=='nan':a[key].flat[0]=np.nan
    if kind=='inf':a[key].flat[0]=np.inf
    if kind=='poor':a[key]*=0
    if kind!='missing':np.savez(target,**a)
    report,code=judge.evaluate(target);assert code=={'good':0,'poor':1}.get(kind,2)
    if code==2:assert report['metrics']=={} and report['errors']
    json.dumps(report,allow_nan=False)

@pytest.mark.parametrize('kind',['missing_reference','zero_reference','bad_boundary','bad_reference_shape','nonfinite_reference'])
def test_evaluator_errors(tmp_path,kind):
    cfg=json.loads((ROOT/'evaluation/metrics.json').read_text());a=arrays()
    if kind=='bad_boundary':cfg['outputs']['reconstruction']['boundary']=-1
    if kind=='zero_reference':a['reconstruction']*=0
    if kind=='bad_reference_shape':a['reconstruction']=a['reconstruction'].ravel()
    if kind=='nonfinite_reference':a['reconstruction'].flat[0]=np.nan
    (tmp_path/'metrics.json').write_text(json.dumps(cfg))
    if kind!='missing_reference':np.savez(tmp_path/'baseline_reference.npz',**a)
    report,code=judge.evaluate(ROOT/'evaluation/baseline_reference.npz',tmp_path)
    assert code==2 and report['metrics']=={} and report['errors']

def test_truth_takes_precedence(tmp_path):
    shutil.copy2(ROOT/'evaluation/metrics.json',tmp_path/'metrics.json');a=arrays();np.savez(tmp_path/'ground_truth.npz',**a)
    a['reconstruction']*=0;np.savez(tmp_path/'baseline_reference.npz',**a)
    assert judge.evaluate(ROOT/'evaluation/baseline_reference.npz',tmp_path)[1]==0

def test_cli_foreign_cwd_and_report_failure(tmp_path):
    target=tmp_path/'nested/report.json';cmd=[sys.executable,str(ROOT/'evaluation/judge.py'),'--prediction',str(ROOT/'evaluation/baseline_reference.npz'),'--report',str(target)]
    assert subprocess.run(cmd,cwd=tmp_path,capture_output=True).returncode==0
    assert json.loads(target.read_text())['passed']
    blocker=tmp_path/'blocked';blocker.write_text('file');cmd[-1]=str(blocker/'report.json')
    run=subprocess.run(cmd,cwd=tmp_path,capture_output=True);assert run.returncode==2 and b'Cannot write report' in run.stderr

@pytest.mark.parametrize('key',list(arrays()))
def test_each_quantity_is_required_and_scored(tmp_path,key):
    a=arrays();a[key]*=0;path=tmp_path/'one_bad.npz';np.savez(path,**a)
    assert judge.evaluate(path)[1]==1
    a.pop(key);np.savez(path,**a);report,code=judge.evaluate(path)
    assert code==2 and not report['metrics']
