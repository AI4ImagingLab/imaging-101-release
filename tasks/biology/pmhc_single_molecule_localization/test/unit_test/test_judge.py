from pathlib import Path
import importlib.util,subprocess,sys,json,shutil
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('judge',ROOT/'evaluation/judge.py');j=importlib.util.module_from_spec(spec);spec.loader.exec_module(j)

def test_known_assignment_metrics():
    ref=np.array([[0,1,2,10,2,.1],[0,4,5,20,3,.2]],float)
    assert j.score(ref[::-1],ref)['detection_f1']==1.
    x=ref.copy();x[:,1]+=.1;x[:,3]*=1.2
    m=j.score(x,ref);assert m['position_rmse_pixels']==pytest.approx(.1);assert m['mass_relative_rmse']==pytest.approx(.2)
    assert j.score(ref[:1],ref)['detection_f1']==pytest.approx(2/3)

def invoke(pred,report,cwd,script=None):
    p=subprocess.run([sys.executable,str(script or ROOT/'evaluation/judge.py'),'--prediction',str(pred),'--report',str(report)],cwd=cwd,capture_output=True,text=True)
    return p,json.loads(report.read_text()) if report.is_file() else None

def test_contracts(tmp_path):
    x=np.load(ROOT/'evaluation/baseline_reference.npz')['reconstruction'];pred=tmp_path/'p.npz';report=tmp_path/'parent/r.json'
    np.savez(pred,reconstruction=x[:,::-1]);p,r=invoke(pred,report,tmp_path);assert p.returncode==0 and r['passed']
    np.savez(pred,reconstruction=x[:,:0]);p,r=invoke(pred,report,tmp_path);assert p.returncode==1 and r['status']=='fail'
    badframe=x.copy();badframe[0,0,0]=.5
    for kwargs in ({'wrong':x},{'reconstruction':x[0]},{'reconstruction':x.astype('float32')},{'reconstruction':x.astype('complex128')},{'reconstruction':np.full_like(x,np.nan)},{'reconstruction':np.full_like(x,np.inf)},{'reconstruction':badframe}):
        np.savez(pred,**kwargs);p,r=invoke(pred,report,tmp_path);assert p.returncode==2 and r['metrics']=={} and r['errors']
    p,r=invoke(tmp_path/'absent.npz',report,tmp_path);assert p.returncode==2
    p,r=invoke(ROOT/'evaluation/baseline_reference.npz',tmp_path,tmp_path);assert p.returncode==2 and 'Report write failed' in p.stderr

def test_evaluator_failure_truth_priority(tmp_path):
    folder=tmp_path/'evaluation';folder.mkdir()
    for name in ('judge.py','metrics.json'):shutil.copy(ROOT/'evaluation'/name,folder/name)
    report=tmp_path/'r.json';p,r=invoke(ROOT/'evaluation/baseline_reference.npz',report,tmp_path,folder/'judge.py');assert p.returncode==2 and r['metrics']=={}
    shutil.copy(ROOT/'evaluation/baseline_reference.npz',folder/'baseline_reference.npz');np.savez(folder/'ground_truth.npz',wrong=np.zeros(1))
    p,r=invoke(ROOT/'evaluation/baseline_reference.npz',report,tmp_path,folder/'judge.py');assert p.returncode==2 and 'Missing reconstruction' in r['errors'][0]
