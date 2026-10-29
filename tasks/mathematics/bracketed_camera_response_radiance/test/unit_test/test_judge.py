from pathlib import Path
import importlib.util,json,subprocess,sys
import numpy as np
import pytest
T=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('standalone_judge',T/'evaluation/judge.py');judge=importlib.util.module_from_spec(spec);spec.loader.exec_module(judge)

@pytest.fixture
def reference():
    with np.load(T/'evaluation/baseline_reference.npz',allow_pickle=False) as f:return {k:f[k].copy() for k in f.files}

def test_reference_passes():
    report,code=judge.evaluate(T/'evaluation/baseline_reference.npz')
    assert code==0 and report['passed'] and report['metrics']['response_log_curvature']<.0017575872742814459

@pytest.mark.parametrize('kind',['missing_key','extra_key','shape','dtype','nan','inf','complex'])
def test_malformed_predictions(tmp_path,reference,kind):
    if kind=='missing_key':reference.pop('response')
    elif kind=='extra_key':reference['secret']=np.ones(1)
    elif kind=='shape':reference['response']=reference['response'][0]
    elif kind=='dtype':reference['response']=reference['response'].astype(np.float32)
    elif kind=='complex':reference['response']=reference['response'].astype(complex)
    else:reference['response'][0,0,0]=np.nan if kind=='nan' else np.inf
    p=tmp_path/'bad.npz';np.savez(p,**reference);r,c=judge.evaluate(p)
    assert c==2 and r['status']=='error' and r['metrics']=={} and r['errors']

@pytest.mark.parametrize('kind',['zero','constant','gain','response','gauge'])
def test_poor_predictions_fail(tmp_path,reference,kind):
    if kind=='zero':reference['reconstruction']*=0
    elif kind=='constant':reference['reconstruction'][:]=1
    elif kind=='gain':reference['reconstruction']*=2
    elif kind=='response':reference['response']**=1.5
    else:
        reference['response']*=1.00001;reference['reconstruction']*=1.00001
    p=tmp_path/'poor.npz';np.savez(p,**reference);r,c=judge.evaluate(p)
    assert c==1 and not r['passed'] and r['status']=='fail'

def test_missing_file():
    r,c=judge.evaluate(T/'missing.npz');assert c==2 and r['metrics']=={}

def test_evaluator_failure(tmp_path):
    r,c=judge.evaluate(T/'evaluation/baseline_reference.npz',tmp_path);assert c==2 and r['metrics']=={}

def test_foreign_cwd_cli_and_report_failure(tmp_path):
    p=tmp_path/'nested/report.json'
    cmd=[sys.executable,str(T/'evaluation/judge.py'),'--prediction',str(T/'evaluation/baseline_reference.npz'),'--report',str(p)]
    run=subprocess.run(cmd,cwd=tmp_path,capture_output=True);assert run.returncode==0 and json.loads(p.read_text())['passed']
    run=subprocess.run(cmd[:-1]+[str(tmp_path)],cwd=tmp_path,capture_output=True);assert run.returncode==2 and b'Cannot write report' in run.stderr
