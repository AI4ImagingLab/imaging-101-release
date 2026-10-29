from pathlib import Path
import importlib.util,sys,subprocess,json,shutil
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from src.preprocessing import load_data,preprocess
from src.physics_model import instrument_lineshape,optical_depth,forward
from src.solvers import INITIAL
spec=importlib.util.spec_from_file_location('judge',ROOT/'evaluation/judge.py');judge=importlib.util.module_from_spec(spec);spec.loader.exec_module(judge)
def test_physics():
    d=load_data(ROOT/'data/raw_data.npz');g=instrument_lineshape(.6,2,0,0)
    assert (g>=0).all() and abs(sum(g)-1)<1e-14
    np.testing.assert_allclose(np.convolve(np.ones(1201),g,'same')[200:-200],1,atol=1e-14)
    p=INITIAL.copy();p[:3]=0;od=optical_depth(p,d['cross_sections']);assert np.array_equal(od,np.zeros_like(od))
    p[0]=1e18;a=optical_depth(p,d['cross_sections']);p[0]=2e18;b=optical_depth(p,d['cross_sections']);np.testing.assert_allclose(b,2*a)
    assert np.all(np.exp(-b[0])<=np.exp(-a[0]))
    x,_=preprocess(d['wavelength'],d['spectra'][0],d['dark'])
    p=INITIAL.copy();p[0]=5e17;f=forward(x,p,d['model_wavelength'],d['solar'],d['cross_sections']);p[0]+=1e14;f2=forward(x,p,d['model_wavelength'],d['solar'],d['cross_sections'])
    assert np.linalg.norm(f2-f)>0 and np.mean(f2)<np.mean(f)
def test_metrics_analytic():
    a=np.array([0.,1.,2.]);m=judge.score(a,a);assert abs(m['ncc']-1)<1e-14 and m['nrmse']==0
    m=judge.score(2*a,a);assert abs(m['ncc']-1)<1e-14 and abs(m['nrmse']-np.sqrt(5/3)/2)<1e-14
    assert judge.score(np.zeros(3),a)['ncc']==0
    with pytest.raises(ValueError):judge.score(a,np.ones(3))
def profile():
    with np.load(ROOT/'evaluation/baseline_reference.npz') as z:return z['reconstruction'].copy()
@pytest.mark.parametrize('kind',['missing','key','shape','dtype','complex','nan','inf','object'])
def test_malformed(tmp_path,kind):
    p=tmp_path/'bad.npz';a=profile()
    if kind=='key':np.savez(p,wrong=a)
    elif kind!='missing':
        if kind=='shape':a=a[0]
        if kind=='dtype':a=np.zeros((1,360),dtype=int)
        if kind=='complex':a=a.astype(complex)
        if kind=='nan':a[0,0]=np.nan
        if kind=='inf':a[0,0]=np.inf
        if kind=='object':a=a.astype(object)
        np.savez(p,reconstruction=a)
    r,c=judge.evaluate(p);assert c==2 and r['metrics']=={} and r['errors']
@pytest.mark.parametrize('kind',['zero','scale','reverse','constant'])
def test_bad_quality(tmp_path,kind):
    a=profile()
    if kind=='zero':a*=0
    if kind=='scale':a*=2
    if kind=='reverse':a=a[:,::-1]
    if kind=='constant':a[:]=np.mean(a)
    p=tmp_path/'bad.npz';np.savez(p,reconstruction=a);assert judge.evaluate(p)[1]==1
def test_reference_precedence_and_errors(tmp_path,monkeypatch):
    e=tmp_path/'evaluation';e.mkdir();shutil.copyfile(ROOT/'evaluation/metrics.json',e/'metrics.json')
    a=profile();p=tmp_path/'good.npz';np.savez(p,reconstruction=a)
    np.savez(e/'ground_truth.npz',reconstruction=a);np.savez(e/'baseline_reference.npz',reconstruction=2*a)
    monkeypatch.setattr(judge,'HERE',e);assert judge.evaluate(p)[1]==0
    np.savez(e/'ground_truth.npz',reconstruction=np.ones((1,360)));assert judge.evaluate(p)[1]==2
    (e/'ground_truth.npz').unlink();(e/'baseline_reference.npz').unlink();assert judge.evaluate(p)[1]==2
def test_cli_foreign_cwd_and_write_error(tmp_path):
    p=tmp_path/'good.npz';np.savez(p,reconstruction=profile());r=tmp_path/'new'/'report.json'
    cmd=[sys.executable,str(ROOT/'evaluation/judge.py'),'--prediction',str(p),'--report']
    assert subprocess.run(cmd+[str(r)],cwd=tmp_path,capture_output=True).returncode==0
    assert json.loads(r.read_text())['passed']
    assert subprocess.run(cmd+[str(tmp_path)],capture_output=True).returncode==2
def test_public_contract():
    text=(ROOT/'README.md').read_text(encoding='utf8');d=load_data(ROOT/'data/raw_data.npz')
    for k in d:assert k in text
    for v in ['310','320','280','290','0.01','295','243']:assert v in text
    assert d['spectra'].shape==(360,2048)
