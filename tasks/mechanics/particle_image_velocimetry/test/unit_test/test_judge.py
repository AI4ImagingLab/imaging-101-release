import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('independent_judge',ROOT/'evaluation/judge.py')
judge=importlib.util.module_from_spec(spec);spec.loader.exec_module(judge)

def valid():
    with np.load(ROOT/'evaluation/baseline_reference.npz',allow_pickle=False) as d:return {k:d[k] for k in d.files}

def test_known_metrics():
    a=dict(reconstruction=np.array([0.,2.]),confidence=np.array([1.,2.]))
    b=dict(reconstruction=np.array([0.,1.]),confidence=np.array([1.,2.]))
    m=judge.score(a,b)
    assert m['ncc']==1 and m['nrmse']==np.sqrt(.5) and m['confidence_relative_l2']==0

def test_valid_and_poor(tmp_path):
    p=tmp_path/'a.npz';d=valid();np.savez(p,**d)
    assert judge.evaluate(p)[1]==0
    d['reconstruction']=np.zeros_like(d['reconstruction']);np.savez(p,**d)
    report,code=judge.evaluate(p);assert code==1 and report['status']=='fail'

@pytest.mark.parametrize('mode',['missing','key','shape','int','float32','complex','nan','inf','object','confidence','overflow'])
def test_invalid(tmp_path,mode):
    p=tmp_path/'bad.npz';d=valid()
    if mode=='key':del d['reconstruction']
    if mode=='shape':d['reconstruction']=d['reconstruction'][0]
    if mode=='int':d['reconstruction']=d['reconstruction'].astype(int)
    if mode=='float32':d['reconstruction']=d['reconstruction'].astype(np.float32)
    if mode=='complex':d['reconstruction']=d['reconstruction'].astype(complex)
    if mode in ('nan','inf'):d['reconstruction'].flat[0]=float(mode)
    if mode=='object':d['reconstruction']=d['reconstruction'].astype(object)
    if mode=='confidence':d['confidence']=np.zeros((1,1))
    if mode=='overflow':d['reconstruction'][:]=1e308
    if mode!='missing':np.savez(p,**d)
    r,code=judge.evaluate(p)
    assert code==2 and r['status']=='error' and r['metrics']=={} and r['errors']

def test_foreign_cwd_and_report_write(tmp_path):
    p=tmp_path/'p.npz';np.savez(p,**valid());report=tmp_path/'nested/r.json'
    cmd=[sys.executable,str(ROOT/'evaluation/judge.py'),'--prediction',str(p),'--report',str(report)]
    run=subprocess.run(cmd,cwd=tmp_path,capture_output=True,text=True)
    assert run.returncode==0 and json.loads(report.read_text())['passed']
    run=subprocess.run(cmd[:-1]+[str(tmp_path)],cwd=tmp_path,capture_output=True,text=True)
    assert run.returncode==2 and 'Cannot write report' in run.stderr

def test_unavailable_hidden_assets(tmp_path,monkeypatch):
    monkeypatch.setattr(judge,'__file__',str(tmp_path/'judge.py'))
    r,code=judge.evaluate('irrelevant.npz');assert code==2 and r['metrics']=={}

def test_degenerate_reference_metrics_rejected():
    d=dict(reconstruction=np.ones((1,7,10,2)),confidence=np.ones((1,7,10)))
    with pytest.raises(ValueError,match='Degenerate'):judge.score(d,d)

def test_invalid_metric_configuration(tmp_path,monkeypatch):
    monkeypatch.setattr(judge,'__file__',str(tmp_path/'judge.py'))
    (tmp_path/'metrics.json').write_text('{"ncc_boundary": null, "nrmse_boundary": 0.1, "confidence_relative_l2_boundary": 0.1}')
    r,code=judge.evaluate('irrelevant.npz')
    assert code==2 and r['metrics']=={} and 'Invalid metric boundary' in r['errors'][0]
