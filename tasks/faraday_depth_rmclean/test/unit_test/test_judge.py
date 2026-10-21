import importlib.util,json,subprocess,sys
from pathlib import Path
import numpy as np
import pytest

TASK=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('judge',TASK/'evaluation/judge.py')
judge=importlib.util.module_from_spec(spec);spec.loader.exec_module(judge)

def test_known_complex_relative_l2():
    assert judge.relative_l2(np.array([2+2j]),np.array([1+1j]))==1.
    assert judge.relative_l2(np.array([1j,1]),np.array([1j,1]))==0.

@pytest.mark.parametrize('case',['missing','key','shape','dtype','nan','inf','overflow','zero','wrongphase'])
def test_invalid_and_poor_submissions(tmp_path,case):
    prediction=tmp_path/'prediction.npz'
    with np.load(TASK/'evaluation/baseline_reference.npz',allow_pickle=False) as f:
        arrays={key:f[key].copy() for key in f.files}
    expected=2
    if case=='key':arrays.pop('components')
    elif case=='shape':arrays['components']=arrays['components'][0]
    elif case=='dtype':arrays['components']=arrays['components'].real
    elif case in ('nan','inf'):arrays['components'][0,0]=float(case)
    elif case=='overflow':arrays['components'][:]=1e308+1e308j
    elif case=='zero':arrays={key:np.zeros_like(x) for key,x in arrays.items()};expected=1
    elif case=='wrongphase':arrays={key:x*1j for key,x in arrays.items()};expected=1
    if case!='missing':np.savez_compressed(prediction,**arrays)
    report,status=judge.evaluate(prediction)
    assert status==expected and not report['passed']
    assert report['status']==('error' if expected==2 else 'fail')
    if expected==2:assert report['metrics']=={} and report['errors']
    json.dumps(report,allow_nan=False)

def test_reference_passes_and_cli_foreign_cwd(tmp_path):
    report=tmp_path/'nested/report.json'
    p=subprocess.run([sys.executable,str(TASK/'evaluation/judge.py'),'--prediction',str(TASK/'evaluation/baseline_reference.npz'),'--report',str(report)],cwd=tmp_path,capture_output=True,text=True)
    assert p.returncode==0,p.stderr
    data=json.loads(report.read_text(encoding='utf8'));assert data['passed']

@pytest.mark.parametrize('boundary',[True,0.,float('nan'),float('inf'),-1])
def test_bad_metric_configuration(tmp_path,monkeypatch,boundary):
    old=judge.METRICS
    d=json.loads(old.read_text(encoding='utf8'));d['components_relative_l2_boundary']=boundary
    path=tmp_path/'metrics.json';path.write_text(json.dumps(d),encoding='utf8')
    monkeypatch.setattr(judge,'METRICS',path)
    r,code=judge.evaluate(TASK/'evaluation/baseline_reference.npz');assert code==2 and r['metrics']=={}

def test_missing_hidden_reference(tmp_path,monkeypatch):
    monkeypatch.setattr(judge,'HIDDEN',tmp_path)
    r,c=judge.evaluate(TASK/'evaluation/baseline_reference.npz');assert c==2 and r['metrics']=={}

def test_report_write_failure(tmp_path):
    p=subprocess.run([sys.executable,str(TASK/'evaluation/judge.py'),'--prediction',str(TASK/'evaluation/baseline_reference.npz'),'--report',str(tmp_path)],capture_output=True,text=True)
    assert p.returncode==2 and 'report' in p.stderr.lower()
