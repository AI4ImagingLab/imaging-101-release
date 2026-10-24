from pathlib import Path
import importlib.util,json,subprocess,sys
import numpy as np
import pytest
T=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('independent_judge',T/'evaluation/judge.py');judge=importlib.util.module_from_spec(spec);spec.loader.exec_module(judge)

@pytest.fixture
def evaluator(tmp_path,monkeypatch):
    root=tmp_path/'evaluation';root.mkdir()
    for name in ['metrics.json','baseline_reference.npz']:(root/name).write_bytes((T/'evaluation'/name).read_bytes())
    monkeypatch.setattr(judge,'HERE',root)
    monkeypatch.setattr(judge,'observation_context',lambda:None)
    monkeypatch.setattr(judge,'forward_error',lambda x,c:0.)
    return root

@pytest.mark.parametrize('case',['missing','key','extra','shape','int','float32','complex','nan','inf','object'])
def test_invalid_submissions(evaluator,tmp_path,case):
    a=np.ones((1,5707),dtype=np.float64);prediction=tmp_path/'prediction.npz'
    if case=='missing':pass
    elif case=='key':np.savez(prediction,wrong=a)
    elif case=='extra':np.savez(prediction,reconstruction=a,extra=a)
    else:
        if case=='shape':a=a.T
        if case=='int':a=a.astype(np.int64)
        if case=='float32':a=a.astype(np.float32)
        if case=='complex':a=a.astype(np.complex128)
        if case=='nan':a[0,0]=np.nan
        if case=='inf':a[0,0]=np.inf
        if case=='object':a=a.astype(object)
        np.savez(prediction,reconstruction=a)
    r=judge.evaluate(prediction);assert r['status']=='error' and r['metrics']=={} and r['errors'];json.dumps(r,allow_nan=False)

def test_actual_truth_priority_and_invalid_truth_no_fallback(evaluator,tmp_path):
    truth=np.linspace(1e-6,2e-5,5707)[None];prediction=tmp_path/'prediction.npz';np.savez(evaluator/'ground_truth.npz',reconstruction=truth);np.savez(prediction,reconstruction=truth)
    assert judge.evaluate(prediction)['passed']
    np.savez(evaluator/'ground_truth.npz',reconstruction=np.ones((1,5707)))
    r=judge.evaluate(prediction);assert r['status']=='error' and r['metrics']=={} and 'Evaluator' in r['errors'][0]

def test_zero_is_valid_failure_and_one_outlier_fails(evaluator,tmp_path):
    pred=tmp_path/'prediction.npz';np.savez(pred,reconstruction=np.zeros((1,5707)))
    r=judge.evaluate(pred);assert r['status']=='fail' and r['metrics']['yield_ncc']==0
    a=np.load(evaluator/'baseline_reference.npz')['reconstruction'];a[0,0]+=2e-10;np.savez(pred,reconstruction=a)
    assert judge.evaluate(pred)['status']=='fail'

@pytest.mark.parametrize('value',[None,True,float('nan'),-1])
def test_invalid_threshold(evaluator,tmp_path,value):
    p=evaluator/'metrics.json';cfg=json.loads(p.read_text());cfg['nrmse_boundary']=value;p.write_text(json.dumps(cfg));r=judge.evaluate(tmp_path/'missing.npz');assert r['status']=='error' and not r['metrics']

def test_known_metric_definitions():
    x=np.array([2.,4.]);y=np.array([1.,3.]);m=judge.field_metrics(x,y)
    assert m['yield_ncc']==pytest.approx(14/np.sqrt(20)/np.sqrt(10))
    assert m['yield_nrmse']==.5 and m['yield_max_abs']==1
    assert judge.field_metrics(np.zeros(2),y)['yield_ncc']==0
    with pytest.raises(ValueError):judge.field_metrics(x,np.ones(2))

def test_no_solver_import_or_backend_change():
    s=(T/'evaluation/judge.py').read_text();assert 'from src' not in s and 'import src' not in s
    import matplotlib
    original=matplotlib.get_backend();import src.main
    assert matplotlib.get_backend()==original


def test_cli_exit_codes_and_report_write_failure(tmp_path):
    good=T/'evaluation/baseline_reference.npz';poor=tmp_path/'poor.npz';np.savez(poor,reconstruction=np.zeros((1,5707)))
    for prediction,expected in [(good,0),(poor,1),(tmp_path/'missing.npz',2)]:
        report=tmp_path/f'report{expected}.json';r=subprocess.run([sys.executable,str(T/'evaluation/judge.py'),'--prediction',str(prediction),'--report',str(report)],cwd=tmp_path,capture_output=True,text=True)
        assert r.returncode==expected,(r.stdout,r.stderr);data=json.loads(report.read_text());assert data['status']==['pass','fail','error'][expected]
    blocked=tmp_path/'blocked';blocked.write_text('file');r=subprocess.run([sys.executable,str(T/'evaluation/judge.py'),'--prediction',str(good),'--report',str(blocked/'report.json')],cwd=tmp_path,capture_output=True,text=True)
    assert r.returncode==2 and 'Report write failed' in r.stderr
