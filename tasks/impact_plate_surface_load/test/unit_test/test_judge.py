from pathlib import Path
import importlib.util,numpy as np,pytest,subprocess,sys
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('standalone_judge',ROOT/'evaluation/judge.py');judge=importlib.util.module_from_spec(spec);spec.loader.exec_module(judge)

def test_known_cosine_and_dynamic_range_rmse():
    n,r=judge.ncc_nrmse(np.array([0.,1.,0.]),np.array([0.,1.,2.]));assert n==pytest.approx(1/np.sqrt(5));assert r==pytest.approx(np.sqrt(4/3)/2)

@pytest.mark.parametrize('mode',['missing_file','missing_key','shape','dtype','complex','nan','inf'])
def test_invalid_contract(tmp_path,mode):
    p=tmp_path/'bad.npz'
    if mode!='missing_file':
        a=np.ones(judge.SHAPE,dtype=np.float64)
        if mode=='shape':a=a[...,:-1]
        if mode=='dtype':a=a.astype(np.float32)
        if mode=='complex':a=a.astype(np.complex128)
        if mode=='nan':a.flat[0]=np.nan
        if mode=='inf':a.flat[0]=np.inf
        np.savez(p,**({'wrong':a} if mode=='missing_key' else {'reconstruction':a}))
    r,c=judge.evaluate(p);assert c==2 and r['metrics']=={} and r['errors']

def test_bad_quality_and_independent_force_criterion(tmp_path):
    p=tmp_path/'bad.npz';np.savez(p,reconstruction=np.ones(judge.SHAPE));r,c=judge.evaluate(p);assert c==1 and not r['passed'] and r['metrics']['force_nrmse']>r['thresholds']['force_nrmse']['boundary']

def test_unavailable_evaluator_assets(tmp_path,monkeypatch):
    p=tmp_path/'valid.npz';np.savez(p,reconstruction=np.ones(judge.SHAPE));monkeypatch.setattr(judge,'ROOT',tmp_path);r,c=judge.evaluate(p);assert c==2 and r['metrics']=={}

def test_report_write_error(tmp_path):
    r=subprocess.run([sys.executable,str(ROOT/'evaluation/judge.py'),'--prediction',str(tmp_path/'none'),'--report',str(tmp_path)],capture_output=True,text=True);assert r.returncode==2 and 'Cannot write report' in r.stderr


@pytest.mark.parametrize('key,value', [('ncc_boundary',float('nan')),('ncc_boundary',float('inf')),('ncc_boundary',None),('ncc_boundary',True),('ncc_boundary',1.1),('ncc_boundary',-1.1),('nrmse_boundary',-1),('force_nrmse_boundary',-1),('force_nrmse_boundary','0.1'),('force_nrmse_boundary',False),('nrmse_boundary','MISSING')])
def test_invalid_threshold_configuration(tmp_path,monkeypatch,key,value):
    import shutil,json
    shutil.copyfile(ROOT/'evaluation/baseline_reference.npz',tmp_path/'baseline_reference.npz')
    cfg=json.loads((ROOT/'evaluation/metrics.json').read_text())
    if value=='MISSING':cfg.pop(key)
    else:cfg[key]=value
    (tmp_path/'metrics.json').write_text(json.dumps(cfg))
    with np.load(tmp_path/'baseline_reference.npz') as f:a=f['reconstruction']
    p=tmp_path/'valid.npz';np.savez(p,reconstruction=a);monkeypatch.setattr(judge,'ROOT',tmp_path)
    r,c=judge.evaluate(p);assert c==2 and r['metrics']=={} and r['errors']


def test_finite_extreme_values_return_error(tmp_path):
    p=tmp_path/'extreme.npz';np.savez(p,reconstruction=np.full(judge.SHAPE,1e308))
    r,c=judge.evaluate(p);assert c==2 and r['metrics']=={} and r['errors']


def test_tightened_valid_threshold_takes_effect(tmp_path,monkeypatch):
    import shutil,json
    shutil.copyfile(ROOT/'evaluation/baseline_reference.npz',tmp_path/'baseline_reference.npz')
    cfg=json.loads((ROOT/'evaluation/metrics.json').read_text());cfg['force_nrmse_boundary']=0.
    (tmp_path/'metrics.json').write_text(json.dumps(cfg))
    with np.load(tmp_path/'baseline_reference.npz') as f:a=f['reconstruction']
    p=tmp_path/'valid.npz';np.savez(p,reconstruction=a);monkeypatch.setattr(judge,'ROOT',tmp_path)
    r,c=judge.evaluate(p);assert c==1 and not r['passed'] and r['metrics']['force_nrmse']>0
