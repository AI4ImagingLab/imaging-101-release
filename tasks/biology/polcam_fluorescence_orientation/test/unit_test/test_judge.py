from pathlib import Path
import importlib.util,numpy as np,pytest,subprocess,sys,shutil,json
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('standalone',ROOT/'evaluation/judge.py');judge=importlib.util.module_from_spec(spec);spec.loader.exec_module(judge)

def valid():
    with np.load(ROOT/'evaluation/baseline_reference.npz',allow_pickle=False) as f:return {k:f[k] for k in judge.SHAPES}

def test_metric_known_case():
    # Dynamic range is 2, RMSE=sqrt(2/3), cosine=1/sqrt(5).
    n,r=judge.ncc_nrmse(np.array([0.,1.,0.]),np.array([0.,1.,2.]));assert n==pytest.approx(1/np.sqrt(5));assert r==pytest.approx(np.sqrt(4/3)/2)

def test_missing_file(tmp_path):
    r,c=judge.evaluate(tmp_path/'missing.npz');assert c==2 and r['metrics']=={} and r['errors']

@pytest.mark.parametrize('mode',['missing','shape','dtype','complex','nan','inf'])
def test_invalid_contract(tmp_path,mode):
    a=valid()
    if mode=='missing':del a['degree']
    elif mode=='shape':a['degree']=a['degree'][...,:-1]
    elif mode=='dtype':a['degree']=a['degree'].astype(np.float32)
    elif mode=='complex':a['degree']=a['degree'].astype(np.complex128)
    elif mode=='nan':a['degree'][0,0,0,0]=np.nan
    elif mode=='inf':a['degree'][0,0,0,0]=np.inf
    p=tmp_path/'invalid.npz';np.savez(p,**a);r,c=judge.evaluate(p);assert c==2 and r['metrics']=={} and r['errors']

@pytest.mark.parametrize('field',['reconstruction','orientation','degree'])
def test_every_required_quantity_rejects_poor_result(tmp_path,field):
    a=valid();a[field]=a[field]+(0.2 if field!='reconstruction' else 100.)
    p=tmp_path/'poor.npz';np.savez(p,**a);r,c=judge.evaluate(p);assert c==1 and not r['passed'] and r['status']=='fail'

def test_evaluator_assets_error(tmp_path,monkeypatch):
    monkeypatch.setattr(judge,'ROOT',tmp_path);r,c=judge.evaluate(ROOT/'evaluation/baseline_reference.npz');assert c==2 and r['metrics']=={} and r['errors']

def test_report_write_error(tmp_path):
    p=subprocess.run([sys.executable,str(ROOT/'evaluation/judge.py'),'--prediction',str(tmp_path/'none.npz'),'--report',str(tmp_path)],capture_output=True,text=True)
    assert p.returncode==2 and 'Cannot write report' in p.stderr
