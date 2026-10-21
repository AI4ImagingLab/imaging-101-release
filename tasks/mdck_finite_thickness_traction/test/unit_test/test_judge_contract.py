from pathlib import Path
import importlib.util,subprocess,sys,json
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[2]
s=importlib.util.spec_from_file_location('j',ROOT/'evaluation/judge.py');j=importlib.util.module_from_spec(s);s.loader.exec_module(j)

def test_known_metrics():
    r=np.array([1.,2.,3.]);assert j.metrics(r,r)=={'ncc':1.,'nrmse':0.}
    assert j.metrics(np.zeros(3),r)['ncc']==0
    assert np.isclose(j.metrics(r+1,r)['nrmse'],.5)
    assert j.metrics(-r,r)['ncc']==-1
    with pytest.raises(ValueError):j.metrics(r,np.ones(3))

@pytest.mark.parametrize('fault',['missing','key','shape','float32','complex','nan','inf','zeros','sign'])
def test_bad_prediction(tmp_path,fault):
    p=tmp_path/'p.npz';r=np.load(ROOT/'evaluation/baseline_reference.npz')['reconstruction'].copy();key='reconstruction';code=2
    if fault=='key':key='bad'
    elif fault=='shape':r=r[:,:,:,:1]
    elif fault=='float32':r=r.astype('float32')
    elif fault=='complex':r=r.astype('complex128')
    elif fault in ('nan','inf'):r.flat[0]=float(fault)
    elif fault=='zeros':r*=0;code=1
    elif fault=='sign':r*=-1;code=1
    if fault!='missing':np.savez(p,**{key:r})
    report,status=j.evaluate(p);assert status==code and not report['passed']
    if code==2:assert report['metrics']=={} and report['errors']

def test_external_cwd_report_error_and_assets(tmp_path):
    args=[sys.executable,str(ROOT/'evaluation/judge.py'),'--prediction',str(ROOT/'evaluation/baseline_reference.npz'),'--report',str(tmp_path/'a/r.json')]
    assert subprocess.run(args,cwd=tmp_path,capture_output=True).returncode==0
    args[-1]=str(tmp_path);assert subprocess.run(args,cwd=tmp_path,capture_output=True).returncode==2
    assert j.evaluate(ROOT/'evaluation/baseline_reference.npz',tmp_path)[1]==2

def test_cli_and_backend():
    import matplotlib
    before=matplotlib.get_backend();import src.main
    assert matplotlib.get_backend()==before
    run=subprocess.run([sys.executable,'-m','src.main'],cwd=ROOT,capture_output=True);assert run.returncode==0
    assert j.evaluate(ROOT/'output/reconstruction.npz')[1]==0
