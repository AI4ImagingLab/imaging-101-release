from pathlib import Path
import importlib.util,json,subprocess,sys
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('independent_judge',ROOT/'evaluation/judge.py');judge=importlib.util.module_from_spec(spec);spec.loader.exec_module(judge)

def test_known_metric_and_failure(tmp_path):
    with np.load(ROOT/'evaluation/baseline_reference.npz') as z:a=z['reconstruction']
    p=tmp_path/'prediction.npz';np.savez(p,reconstruction=a+0.25)
    r=judge.evaluate(p);assert r['status']=='fail';assert r['metrics']['field_max_error']==.25
    np.savez(p,reconstruction=a);assert judge.evaluate(p)['status']=='pass'

@pytest.mark.parametrize('kind',['missing','key','shape','dtype','complex','nan','inf'])
def test_malformed(kind,tmp_path):
    p=tmp_path/'bad.npz';a=np.ones((1,5),dtype=np.float64)
    if kind=='key':np.savez(p,other=a)
    elif kind=='shape':np.savez(p,reconstruction=a.T)
    elif kind=='dtype':np.savez(p,reconstruction=a.astype('float32'))
    elif kind=='complex':np.savez(p,reconstruction=a.astype('complex128'))
    elif kind in ['nan','inf']:
        a[0,0]=np.nan if kind=='nan' else np.inf;np.savez(p,reconstruction=a)
    elif kind!='missing':np.savez(p,reconstruction=a)
    r=judge.evaluate(p);assert r['status']=='error';assert r['metrics']=={};assert r['errors']

def test_reference_precedence_and_asset_errors(tmp_path):
    p=tmp_path/'prediction.npz';np.savez(p,reconstruction=np.ones((1,5)))
    assets=tmp_path/'assets';assets.mkdir();(assets/'metrics.json').write_text(json.dumps(dict(field_max_error_boundary=1e-6)))
    assert judge.evaluate(p,assets)['status']=='error'
    np.savez(assets/'baseline_reference.npz',reconstruction=np.zeros((1,5)))
    np.savez(assets/'ground_truth.npz',reconstruction=np.ones((1,5)))
    assert judge.evaluate(p,assets)['status']=='pass'
    (assets/'metrics.json').write_text(json.dumps(dict(field_max_error_boundary=0)))
    assert judge.evaluate(p,assets)['status']=='error'

def test_cli_exit_and_parent_creation(tmp_path):
    report=tmp_path/'new'/'report.json';p=ROOT/'evaluation/baseline_reference.npz'
    result=subprocess.run([sys.executable,str(ROOT/'evaluation/judge.py'),'--prediction',str(p),'--report',str(report)],cwd=tmp_path,capture_output=True)
    assert result.returncode==0;assert json.loads(report.read_text())['passed']
    result=subprocess.run([sys.executable,str(ROOT/'evaluation/judge.py'),'--prediction',str(p),'--report',str(tmp_path)],capture_output=True)
    assert result.returncode==2 and result.stderr

def test_cli_fail_and_invalid_exit_codes(tmp_path):
    p=tmp_path/'prediction.npz';report=tmp_path/'report.json'
    np.savez(p,reconstruction=np.zeros((1,5)))
    args=[sys.executable,str(ROOT/'evaluation/judge.py'),'--prediction',str(p),'--report',str(report)]
    result=subprocess.run(args,capture_output=True);assert result.returncode==1
    np.savez(p,reconstruction=np.zeros((5,1)))
    result=subprocess.run(args,capture_output=True);assert result.returncode==2
