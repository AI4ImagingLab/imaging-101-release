from pathlib import Path
import importlib.util,json,subprocess,sys,shutil
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('independent_judge',ROOT/'evaluation/judge.py')
judge=importlib.util.module_from_spec(spec);spec.loader.exec_module(judge)

def run(tmp,prediction,home=ROOT):
    report=tmp/'new/report.json'
    p=subprocess.run([sys.executable,str(home/'evaluation/judge.py'),'--prediction',str(prediction),'--report',str(report)],cwd=tmp,capture_output=True,text=True,encoding='utf-8')
    assert report.exists(), (p.returncode,p.stdout,p.stderr)
    return p.returncode,json.loads(report.read_text(encoding='utf-8'))

def save(tmp,**arrays):
    p=tmp/'prediction.npz';np.savez(p,**arrays);return p

def good():
    with np.load(ROOT/'evaluation/baseline_reference.npz',allow_pickle=False) as z:return z['reconstruction']

def test_fresh_cli_foreign_cwd_and_float32(tmp_path):
    cli=subprocess.run([sys.executable,'-m','src.main'],cwd=ROOT,capture_output=True,text=True,encoding='utf-8')
    assert cli.returncode==0,cli.stderr
    code,r=run(tmp_path,ROOT/'output/reconstruction.npz')
    assert code==0 and r['passed'] and set(r['metrics'])==set(judge.NAMES)
    assert run(tmp_path,save(tmp_path,reconstruction=good().astype(np.float32)))[0]==0

@pytest.mark.parametrize('kind',['missing','key','shape','int','complex','nan','inf','extra'])
def test_malformed_prediction(tmp_path,kind):
    a=good().copy()
    if kind=='missing':p=tmp_path/'absent.npz'
    elif kind=='key':p=save(tmp_path,wrong=a)
    elif kind=='extra':p=save(tmp_path,reconstruction=a,other=a)
    else:
        if kind=='shape':a=a[0]
        if kind=='int':a=a.astype(np.int64)
        if kind=='complex':a=a.astype(np.complex128)
        if kind in ['nan','inf']:a[0,1]=float(kind)
        p=save(tmp_path,reconstruction=a)
    code,r=run(tmp_path,p)
    assert code==2 and r['status']=='error' and not r['passed'] and r['metrics']=={} and r['errors']

@pytest.mark.parametrize('kind',['zero','scale','negative','prior','reverse','roll','point0'])
def test_bad_scientific_outputs(tmp_path,kind):
    a=good().copy()
    if kind=='zero':a*=0
    if kind=='scale':a*=2
    if kind=='negative':a[0,10]=-1
    if kind=='prior':
        from src.solvers import initial_density
        a=initial_density()[None]
    if kind=='reverse':a=a[:,::-1]
    if kind=='roll':a=np.roll(a,8,axis=1)
    if kind=='point0':a*=0;a[0,0]=1
    code,r=run(tmp_path,save(tmp_path,reconstruction=a))
    assert code==1 and r['status']=='fail' and not r['passed']

@pytest.mark.parametrize('value',[True,False,'0.1',[],[.1],None,float('nan'),float('inf'),-1])
def test_bad_config_boundaries(tmp_path,value):
    p=tmp_path/'metrics.json'
    config=json.loads((ROOT/'evaluation/metrics.json').read_text())
    config['marginal_relative_l2_boundary']=value
    p.write_text(json.dumps(config),encoding='utf-8')
    with pytest.raises(ValueError):judge.configuration(p)

@pytest.mark.parametrize('mode',['empty','missing','extra','comparison'])
def test_bad_config_set(tmp_path,mode):
    config=json.loads((ROOT/'evaluation/metrics.json').read_text())
    if mode=='empty':config={}
    if mode=='missing':config.pop('marginal_relative_l2_boundary')
    if mode=='extra':config['unknown_boundary']=1
    if mode=='comparison':config['comparisons']={}
    p=tmp_path/'metrics.json';p.write_text(json.dumps(config),encoding='utf-8')
    with pytest.raises(ValueError):judge.configuration(p)

def test_evaluator_asset_and_report_failure(tmp_path):
    home=tmp_path/'isolated';(home/'evaluation').mkdir(parents=True)
    shutil.copyfile(ROOT/'evaluation/judge.py',home/'evaluation/judge.py')
    shutil.copyfile(ROOT/'evaluation/metrics.json',home/'evaluation/metrics.json')
    code,r=run(tmp_path,save(tmp_path,reconstruction=good()),home)
    assert code==2 and not r['metrics']
    p=subprocess.run([sys.executable,str(ROOT/'evaluation/judge.py'),'--prediction',str(ROOT/'output/reconstruction.npz'),'--report',str(ROOT/'README.md'/'bad.json')],capture_output=True,text=True,encoding='utf-8')
    assert p.returncode==2 and 'Cannot write' in p.stderr

def test_judge_forward_independent_from_solver():
    from src.physics_model import forward
    from src.preprocessing import load_observations,preprocess
    raw=load_observations(ROOT/'data/raw_data.npz');i,_=preprocess(*raw)
    expected=forward(good()[0],raw[2],np.isfinite(i))['total']
    actual=judge.modeled(good()[0],raw[2]);actual[~np.isfinite(i)]=np.nan
    np.testing.assert_allclose(actual,expected,rtol=1e-11,atol=1e-13,equal_nan=True)
    measured,q=judge.observation(ROOT/'data/raw_data.npz')
    metrics=judge.score(good()[0],measured,q)
    assert abs(metrics['marginal_relative_l2']-.007229869879588262)<1e-12
