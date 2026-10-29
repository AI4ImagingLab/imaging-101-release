import importlib.util,json,subprocess,sys,shutil
from pathlib import Path
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('judge',ROOT/'evaluation/judge.py');judge=importlib.util.module_from_spec(spec);spec.loader.exec_module(judge)

def test_known_metrics():
    y=np.array([0.,1.,2.]);m=judge.component_metrics(y,y)
    assert m['ncc']==pytest.approx(1.) and m['nrmse']==0.
    m=judge.component_metrics(np.array([2.,1.,0.]),y)
    assert m['ncc']==pytest.approx(.2) and m['nrmse']==pytest.approx(np.sqrt(8/3)/2)

def invoke(pred,report,cwd):
    p=subprocess.run([sys.executable,str(ROOT/'evaluation/judge.py'),'--prediction',str(pred),'--report',str(report)],cwd=cwd,capture_output=True,text=True)
    return p,json.loads(report.read_text()) if report.exists() and report.is_file() else None

def test_valid_poor_malformed_and_foreigncwd(tmp_path):
    x=np.load(ROOT/'evaluation/baseline_reference.npz')['reconstruction'];pred=tmp_path/'p.npz';report=tmp_path/'newdir/r.json'
    np.savez(pred,reconstruction=x);p,r=invoke(pred,report,tmp_path);assert p.returncode==0 and r['status']=='pass'
    np.savez(pred,reconstruction=np.zeros_like(x));p,r=invoke(pred,report,tmp_path);assert p.returncode==1 and not r['passed']
    for kwargs in ({'wrong':x},{'reconstruction':x[0]},{'reconstruction':x.astype('float32')},{'reconstruction':x.astype('complex128')},{'reconstruction':np.full_like(x,np.nan)},{'reconstruction':np.full_like(x,np.inf)}):
        np.savez(pred,**kwargs);p,r=invoke(pred,report,tmp_path);assert p.returncode==2 and r['metrics']=={} and r['errors']
    p,r=invoke(tmp_path/'absent.npz',report,tmp_path);assert p.returncode==2 and r['metrics']=={}

def test_evaluator_missing_asset_and_report_write(tmp_path):
    folder=tmp_path/'evaluation';folder.mkdir();shutil.copy(ROOT/'evaluation/judge.py',folder/'judge.py');shutil.copy(ROOT/'evaluation/metrics.json',folder/'metrics.json')
    report=tmp_path/'r.json';p=subprocess.run([sys.executable,str(folder/'judge.py'),'--prediction',str(ROOT/'output/reconstruction.npz'),'--report',str(report)],capture_output=True)
    assert p.returncode==2 and json.loads(report.read_text())['metrics']=={}
    p=subprocess.run([sys.executable,str(ROOT/'evaluation/judge.py'),'--prediction',str(ROOT/'output/reconstruction.npz'),'--report',str(tmp_path)],capture_output=True)
    assert p.returncode==2 and b'Report write failed' in p.stderr

def test_truth_priority(tmp_path):
    folder=tmp_path/'evaluation';folder.mkdir()
    for name in ('judge.py','metrics.json','baseline_reference.npz'):shutil.copy(ROOT/'evaluation'/name,folder/name)
    np.savez(folder/'ground_truth.npz',reconstruction=np.ones((1,68,2,4860)))
    spec=importlib.util.spec_from_file_location('private_judge',folder/'judge.py');j=importlib.util.module_from_spec(spec);spec.loader.exec_module(j)
    r=j.evaluate(ROOT/'evaluation/baseline_reference.npz');assert r['status']=='error' and 'Undefined' in r['errors'][0]

