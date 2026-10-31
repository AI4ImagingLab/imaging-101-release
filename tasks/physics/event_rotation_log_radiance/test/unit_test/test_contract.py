import importlib.util,json,os,subprocess,sys,shutil
from pathlib import Path
import numpy as np,pytest
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('event_judge',ROOT/'evaluation/judge.py');judge=importlib.util.module_from_spec(spec);spec.loader.exec_module(judge)
def run(pred,dest,script=None):return subprocess.run([sys.executable,str(script or ROOT/'evaluation/judge.py'),'--prediction',str(pred),'--report',str(dest)],capture_output=True,text=True)
def test_independently_known_metrics():
    a=np.array([0.,1.,2.]);b=2*a;m=judge.score(b,a);assert abs(m['ncc']-1)<1e-15;assert abs(m['nrmse']-np.sqrt(5/3)/2)<1e-15
    from src.visualization import metrics
    shifted=a+1.;expected=8./np.sqrt(70.)
    assert abs(judge.score(shifted,a)['ncc']-expected)<1e-15
    assert judge.score(shifted,a)==metrics(shifted,a)
    with pytest.raises(ValueError):judge.score(a,np.ones(3))
def test_standard_main_and_foreign_cwd_judge(tmp_path):
    (tmp_path/'data').mkdir();shutil.copyfile(ROOT/'data/raw_data.npz',tmp_path/'data/raw_data.npz');env=os.environ.copy();env['PYTHONPATH']=str(ROOT)
    r=subprocess.run([sys.executable,'-m','src.main'],cwd=tmp_path,env=env,capture_output=True,text=True);assert r.returncode==0,r.stderr
    r=run(tmp_path/'output/reconstruction.npz',tmp_path/'reports/r.json');assert r.returncode==0,r.stderr
    assert json.loads((tmp_path/'reports/r.json').read_text())['passed']
@pytest.mark.parametrize('kind',['missing','key','shape','float32','complex','object','nan','inf','extreme'])
def test_invalid_inputs(tmp_path,kind):
    p=tmp_path/'p.npz';x=np.zeros((1,180,240),dtype=np.float64)
    if kind=='missing':pass
    elif kind=='key':np.savez(p,wrong=x)
    else:
        if kind=='shape':x=x[:,:,:-1]
        if kind=='float32':x=x.astype(np.float32)
        if kind=='complex':x=x.astype(complex)
        if kind=='object':x=x.astype(object)
        if kind=='nan':x.flat[0]=np.nan
        if kind=='inf':x.flat[0]=np.inf
        if kind=='extreme':x.fill(1e308)
        np.savez(p,reconstruction=x)
    dest=tmp_path/'r.json';r=run(p,dest);assert r.returncode==2
    j=json.loads(dest.read_text());assert j['metrics']=={} and j['errors'] and j['status']=='error'
@pytest.mark.parametrize('kind',['zero','inverted','scaled'])
def test_poor_outputs_fail(tmp_path,kind):
    with np.load(ROOT/'evaluation/baseline_reference.npz') as z:x=z['reconstruction'].copy()
    x={'zero':np.zeros_like(x),'inverted':1-x,'scaled':2*x}[kind];p=tmp_path/'p.npz';np.savez(p,reconstruction=x);dest=tmp_path/'r.json';r=run(p,dest);assert r.returncode in (1,2);assert not json.loads(dest.read_text())['passed']
@pytest.mark.parametrize('key,value',[('anisotropic_tv_boundary',True),('anisotropic_tv_boundary',-.1),('observation_residual_boundary',False),('anisotropic_tv_boundary',float('nan')),('anisotropic_tv_boundary',float('inf')),('anisotropic_tv_boundary','bad')])

def test_bad_metric_configuration(tmp_path,monkeypatch,key,value):
    config=json.loads((ROOT/'evaluation/metrics.json').read_text());config[key]=value;(tmp_path/'metrics.json').write_text(json.dumps(config));monkeypatch.setattr(judge,'ASSETS',tmp_path)
    report,code=judge.evaluate(ROOT/'output/reconstruction.npz');assert code==2 and report['metrics']=={} and report['errors']
def test_missing_evaluator_assets(tmp_path,monkeypatch):
    shutil.copyfile(ROOT/'evaluation/metrics.json',tmp_path/'metrics.json');monkeypatch.setattr(judge,'ASSETS',tmp_path);report,code=judge.evaluate('missing.npz');assert code==2 and report['metrics']=={}
def test_write_failure(tmp_path):
    assert run(ROOT/'output/reconstruction.npz',tmp_path).returncode==2
def test_main_import_backend():
    import matplotlib
    old=matplotlib.get_backend();import src.main
    assert matplotlib.get_backend()==old
def test_v2_has_no_hidden_image_dependency(tmp_path):
    assets=tmp_path/'evaluation';assets.mkdir();data=tmp_path/'data';data.mkdir()
    shutil.copyfile(ROOT/'evaluation/judge.py',assets/'judge.py')
    shutil.copyfile(ROOT/'data/raw_data.npz',data/'raw_data.npz')
    shutil.copyfile(ROOT/'evaluation/metrics.json',assets/'metrics.json')
    pred=ROOT/'output/reconstruction.npz';dest=tmp_path/'r.json'
    assert run(pred,dest,assets/'judge.py').returncode==0
    np.savez(assets/'ground_truth.npz',wrong=np.zeros(3))
    assert run(pred,dest,assets/'judge.py').returncode==0
