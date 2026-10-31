import importlib.util,json,subprocess,sys,shutil
from pathlib import Path
import numpy as np
import pytest
from src.preprocessing import load_data,prepare
from src.physics_model import bounded_weights
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('judge',ROOT/'evaluation/judge.py');judge=importlib.util.module_from_spec(spec);spec.loader.exec_module(judge)
def reference():
    with np.load(ROOT/'evaluation/baseline_reference.npz') as z:return {k:z[k] for k in z}
def test_preprocessing_and_metadata():
    raw=load_data(ROOT/'data/raw_data.npz');e,y,s=prepare(raw)
    assert e.shape==(160,) and s.shape==(6,160) and e[0]==11869.66 and e[-1]==12029.756
    assert np.all(np.diff(e)>0)
    readme=(ROOT/'README.md').read_text(encoding='utf8')
    for value in ('11870','12030','11869.660','12029.756'):assert value in readme
def test_bound_transform_corner():
    w=bounded_weights(np.arcsin([.8,.8]),(0,1,2))
    np.testing.assert_allclose(w[:3],[.9,.9,0]);assert w.sum()>1 # native clipping, documented
@pytest.mark.parametrize('kind',['missing','shape','complex','integer','float16','nan','inf','negative','overflow'])
def test_malformed(kind,tmp_path):
    d=reference()
    if kind=='missing':d.pop('weights')
    elif kind=='shape':d['weights']=d['weights'].ravel()
    elif kind=='complex':d['reconstruction']=d['reconstruction'].astype(complex)
    elif kind=='integer':d['reconstruction']=d['reconstruction'].astype(int)
    elif kind=='float16':d['reconstruction']=d['reconstruction'].astype('float16')
    elif kind=='nan':d['reconstruction'][0,0]=np.nan
    elif kind=='inf':d['reconstruction'][0,0]=np.inf
    elif kind=='negative':d['weights'][0,0]=-.1
    elif kind=='overflow':d['reconstruction'][:]=1e308
    p=tmp_path/'bad.npz';np.savez(p,**d);r=judge.evaluate(p)
    assert r['status']=='error' and r['metrics']=={} and r['errors']
@pytest.mark.parametrize('kind',['zero','wrong','shift'])
def test_poor(kind,tmp_path):
    d=reference()
    if kind=='zero':d={k:np.zeros_like(v) for k,v in d.items()}
    elif kind=='wrong':d['weights']=np.array([[1.,0,0,0,0,0]])
    else:d['reconstruction']=np.roll(d['reconstruction'],10,axis=1)
    p=tmp_path/'poor.npz';np.savez(p,**d);assert judge.evaluate(p)['status']=='fail'
def test_metric_known_case_and_float32(tmp_path):
    d=reference();p=tmp_path/'good.npz';np.savez(p,**{k:v.astype('float32') for k,v in d.items()})
    r=judge.evaluate(p);assert r['passed'];assert r['metrics']['weight_max']<1e-7
    d['reconstruction']+=.0001;np.savez(p,**d);r=judge.evaluate(p)
    assert abs(r['metrics']['spectral_rmse']-.0001)<1e-15
def test_missing_file():assert judge.evaluate(ROOT/'absent.npz')['status']=='error'
def test_hidden_failure(monkeypatch,tmp_path):
    monkeypatch.setattr(judge,'HERE',tmp_path);assert judge.evaluate(ROOT/'evaluation/baseline_reference.npz')['status']=='error'
@pytest.mark.parametrize('fault',['reference_negative','reference_nan','boundary_nan','reference_missing'])
def test_evaluator_asset_errors(fault,monkeypatch,tmp_path):
    e=tmp_path/'evaluation';e.mkdir();shutil.copytree(ROOT/'data',tmp_path/'data')
    d=reference();limits=json.loads((ROOT/'evaluation/metrics.json').read_text(encoding='utf8'))
    if fault=='reference_negative':d['weights'][0,0]=-.1
    if fault=='reference_nan':d['reconstruction'][0,0]=np.nan
    if fault=='boundary_nan':limits['weight_max_boundary']=float('nan')
    if fault!='reference_missing':np.savez(e/'baseline_reference.npz',**d)
    (e/'metrics.json').write_text(json.dumps(limits),encoding='utf8');monkeypatch.setattr(judge,'HERE',e)
    r=judge.evaluate(ROOT/'evaluation/baseline_reference.npz');assert r['status']=='error' and r['metrics']=={}
def test_cli_valid_failure(tmp_path):
    p=tmp_path/'bad.npz';np.savez(p,weights=np.zeros((1,6)),reconstruction=np.zeros((1,160)))
    report=tmp_path/'report.json'
    out=subprocess.run([sys.executable,str(ROOT/'evaluation/judge.py'),'--prediction',str(p),'--report',str(report)])
    assert out.returncode==1 and json.loads(report.read_text(encoding='utf8'))['status']=='fail'
def test_cli_pipeline_and_foreign_cwd(tmp_path):
    subprocess.run([sys.executable,'-m','src.main'],cwd=ROOT,check=True)
    p=ROOT/'output/reconstruction.npz';report=tmp_path/'nested/report.json'
    for pred,code in [(p,0),(tmp_path/'missing.npz',2)]:
        r=subprocess.run([sys.executable,str(ROOT/'evaluation/judge.py'),'--prediction',str(pred),'--report',str(report)],cwd=tmp_path)
        assert r.returncode==code;assert json.loads(report.read_text())['status']==('pass' if code==0 else 'error')
    r=subprocess.run([sys.executable,str(ROOT/'evaluation/judge.py'),'--prediction',str(p),'--report',str(tmp_path)],capture_output=True)
    assert r.returncode==2 and b'Report write failed' in r.stderr
