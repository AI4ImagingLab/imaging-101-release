import importlib.util
from pathlib import Path
import numpy as np
import pytest,subprocess,sys,json
spec=importlib.util.spec_from_file_location('judge',Path('evaluation/judge.py'));judge=importlib.util.module_from_spec(spec);spec.loader.exec_module(judge)

@pytest.mark.parametrize('kind',['missing','key','shape','integer','complex','nan','inf','object','zero','negative'])
def test_bad_inputs(tmp_path,kind):
    p=tmp_path/'p.npz';x=np.zeros((1,30,30,13))
    if kind=='missing':pass
    elif kind=='key':np.savez(p,wrong=x)
    else:
        if kind=='shape':x=x[0]
        if kind=='integer':x=x.astype(int)
        if kind=='complex':x=x.astype(complex)
        if kind=='nan':x.flat[0]=np.nan
        if kind=='inf':x.flat[0]=np.inf
        if kind=='object':x=np.array([object()],dtype=object)
        if kind=='negative':x-=1
        np.savez(p,reconstruction=x)
    report,code=judge.evaluate(p)
    assert code==(1 if kind in ('zero','negative') else 2)
    assert not report['passed']
    if code==2:assert report['metrics']=={} and report['errors']
    json.dumps(report,allow_nan=False)

def test_pass_cwd_and_write_failure(tmp_path):
    exe=Path('evaluation/judge.py').resolve();prediction=Path('evaluation/baseline_reference.npz').resolve();report=tmp_path/'nested/r.json'
    p=subprocess.run([sys.executable,str(exe),'--prediction',str(prediction),'--report',str(report)],cwd=tmp_path,capture_output=True)
    assert p.returncode==0 and json.loads(report.read_text())['passed']
    p=subprocess.run([sys.executable,str(exe),'--prediction',str(prediction),'--report',str(tmp_path)],capture_output=True)
    assert p.returncode==2 and b'report write failed' in p.stderr

def test_missing_evaluator_assets(tmp_path):
    report,code=judge.evaluate('output/reconstruction.npz',assets=tmp_path)
    assert code==2 and report['status']=='error' and report['metrics']=={}

def test_independent_known_metric_case():
    a=np.arange(1,11701,dtype=float).reshape(1,30,30,13);b=a*2
    m=np.ones((24,13));y=np.ones((1,30,30,24))
    z=judge.metrics(a,b,y,m)
    np.testing.assert_allclose(z['ncc'],1.)
    np.testing.assert_allclose(z['nrmse'],np.sqrt(np.mean(a*a))/np.ptp(b))


def test_regularized_scale_profiling():
    y=np.ones((1,30,30,24))*128
    m=np.ones((24,13))/13
    x=np.pad(np.ones((1,28,28,13)),((0,0),(1,1),(1,1),(0,0)))
    # Reprofiling the discarded common brightness is exactly scale invariant.
    assert judge.regularized_score(x,y,m)==pytest.approx(judge.regularized_score(x*.37,y,m))
    assert np.isfinite(judge.regularized_score(x,y,m))
