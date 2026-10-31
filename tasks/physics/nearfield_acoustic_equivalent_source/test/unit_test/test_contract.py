from pathlib import Path
import importlib.util,json,subprocess,sys
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('independent_judge',ROOT/'evaluation/judge.py');J=importlib.util.module_from_spec(spec);spec.loader.exec_module(J)
REF=np.load(ROOT/'evaluation/baseline_reference.npz')['reconstruction']
def save(tmp,a,**kwargs):
    p=tmp/'prediction.npz';np.savez(p,reconstruction=a,**kwargs);return p
@pytest.mark.parametrize('factor',[0.,-1.,1j,2.])
def test_reject_bad_scale_phase(tmp_path,factor):
    report,code=J.evaluate(save(tmp_path,(REF*factor).astype(np.complex64)));assert code==1 and report['status']=='fail' and not report['passed']
@pytest.mark.parametrize('kind',['missing','key','shape','dtype','real','nan','inf','object'])
def test_malformed(tmp_path,kind):
    p=tmp_path/'bad.npz';a=REF.copy()
    if kind=='key':np.savez(p,wrong=a)
    elif kind!='missing':
        if kind=='shape':a=a[0]
        if kind=='dtype':a=a.astype(np.complex128)
        if kind=='real':a=a.real.copy()
        if kind=='nan':a[0,0,0]=np.nan
        if kind=='inf':a[0,0,0]=np.inf
        if kind=='object':a=a.astype(object)
        np.savez(p,reconstruction=a)
    report,code=J.evaluate(p);assert code==2 and report['metrics']=={} and report['errors']
def test_known_metrics():
    a=np.array([1+1j,2-1j],dtype=np.complex64);assert J.score(a,a)['relative_complex_l2']==0
    assert J.score(2*a,a)['relative_complex_l2']==1
    assert J.score(1j*a,a)['magnitude_cosine']==pytest.approx(1)
    assert J.score(np.zeros_like(a),a)['magnitude_cosine']==0

def test_regularized_objective_scalar_known_minimum(monkeypatch):
    monkeypatch.setattr(J,'acquisition',lambda:(np.ones((1,1),complex),np.array([3.+0j])))
    weight=3*10**(-35/20)
    optimum=3-weight/np.sqrt(2)
    metric=J.objective_metrics(np.array([[[optimum+0j]]]))
    expected=3*np.sqrt(2)*weight-weight**2/2
    assert metric['regularized_objective']==pytest.approx(expected)
    assert metric['observation_relative_l2']==pytest.approx(weight/(3*np.sqrt(2)))
    assert J.objective_metrics(np.array([[[0j]]]))['regularized_objective']>expected
def test_original_pass_and_finite():
    r,c=J.evaluate(ROOT/'output/reconstruction.npz');assert c==0 and all(np.isfinite(v) for v in r['metrics'].values())
@pytest.mark.parametrize('value',[-1,True,float('inf'),float('nan'),'x'])
def test_invalid_limits(tmp_path,monkeypatch,value):
    assets=tmp_path/'evaluation';assets.mkdir();cfg=json.loads((ROOT/'evaluation/metrics.json').read_text());cfg['regularized_objective_boundary']=value;(assets/'metrics.json').write_text(json.dumps(cfg));monkeypatch.setattr(J,'ASSETS',assets)
    r,c=J.evaluate(ROOT/'output/reconstruction.npz');assert c==2 and not r['metrics']
def test_invalid_cosine_limit(tmp_path,monkeypatch):
    assets=tmp_path/'evaluation';assets.mkdir();cfg=json.loads((ROOT/'evaluation/metrics.json').read_text());cfg['magnitude_cosine_boundary']=1.1;(assets/'metrics.json').write_text(json.dumps(cfg));monkeypatch.setattr(J,'ASSETS',assets)
    r,c=J.evaluate(ROOT/'output/reconstruction.npz');assert c==2
def test_hidden_arrays_do_not_define_v2_target(tmp_path,monkeypatch):
    assets=tmp_path/'evaluation';assets.mkdir();(tmp_path/'data').mkdir();import shutil
    for f in ['metrics.json','baseline_reference.npz']:shutil.copy(ROOT/'evaluation'/f,assets/f)
    shutil.copy(ROOT/'data/raw_data.npz',tmp_path/'data/raw_data.npz');monkeypatch.setattr(J,'ASSETS',assets)
    np.savez(assets/'ground_truth.npz',reconstruction=-REF);r,c=J.evaluate(ROOT/'output/reconstruction.npz');assert c==0
    np.savez(assets/'ground_truth.npz',wrong=REF);r,c=J.evaluate(ROOT/'output/reconstruction.npz');assert c==0
def test_foreign_cwd_report_and_write_error(tmp_path):
    report=tmp_path/'nested'/'report.json';cmd=[sys.executable,str(ROOT/'evaluation/judge.py'),'--prediction',str(ROOT/'output/reconstruction.npz'),'--report',str(report)];r=subprocess.run(cmd,cwd=tmp_path,capture_output=True);assert r.returncode==0 and json.loads(report.read_text())['passed']
    cmd[-1]=str(tmp_path);r=subprocess.run(cmd,cwd=tmp_path,capture_output=True);assert r.returncode==2 and b'Cannot write report' in r.stderr
def test_readme_constants_and_no_public_leak():
    text=(ROOT/'README.md').read_text(encoding='utf-8');assert 'complex64' in text and '(1,40,50)' in text and '1150' in text and '343' in text and '0.025' in text
    for f in ['README.md','requirements.txt']:
        text=(ROOT/f).read_text(encoding='utf-8').lower();assert not any(s in text for s in ['github.com','github.io','githubusercontent','codeload','baseline_reference.npz','ground_truth.npz'])
def test_libraries_do_not_change_backend():
    import matplotlib
    before=matplotlib.get_backend();import src.main,src.visualization
    assert matplotlib.get_backend()==before
