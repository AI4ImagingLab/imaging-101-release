from pathlib import Path
import numpy as np,pytest,importlib.util,subprocess,sys,json
T=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('judge_pore',T/'evaluation/judge.py');judge=importlib.util.module_from_spec(spec);spec.loader.exec_module(judge)
def good():
    with np.load(T/'evaluation/baseline_reference.npz') as f:return {k:f[k].copy() for k in f.files}
def test_good_and_allquantity_quality_failures(tmp_path):
    for mode in ['good','zero','scaled','reversed','density','cumulative']:
        d=good()
        if mode=='zero':d={k:np.zeros_like(v) for k,v in d.items()}
        if mode=='scaled':d={k:2*v for k,v in d.items()}
        if mode=='reversed':d['bin_weights']=d['bin_weights'][:,::-1]
        if mode=='density':d['reconstruction']*=1.1
        if mode=='cumulative':d['cumulative_weights']+=.01
        p=tmp_path/(mode+'.npz');np.savez(p,**d);r,c=judge.evaluate(p);assert c==(0 if mode=='good' else 1),(mode,r)
@pytest.mark.parametrize('mode',['missing','key','shape','float32','integer','nan','inf','complex','negative','pickle'])
def test_invalid_npz(tmp_path,mode):
    d=good();p=tmp_path/'bad.npz'
    if mode!='missing':
        if mode=='key':del d['bin_weights']
        if mode=='shape':d['reconstruction']=d['reconstruction'].ravel()
        if mode=='float32':d['reconstruction']=d['reconstruction'].astype(np.float32)
        if mode=='integer':d['bin_weights']=np.zeros((1,77),dtype=int)
        if mode=='nan':d['bin_weights'][0,0]=np.nan
        if mode=='inf':d['cumulative_weights'][0,0]=np.inf
        if mode=='complex':d['bin_weights']=d['bin_weights'].astype(complex)
        if mode=='negative':d['bin_weights'][0,0]=-1
        if mode=='pickle':d['reconstruction']=np.empty((1,100),dtype=object)
        np.savez(p,**d)
    r,c=judge.evaluate(p);assert c==2 and r['metrics']=={} and r['errors']
def test_evaluator_assets_caller_paths_and_writeerror(tmp_path):
    p=tmp_path/'prediction.npz';np.savez(p,**good());r,c=judge.evaluate(p,tmp_path);assert c==2 and not r['metrics']
    cmd=[sys.executable,str(T/'evaluation/judge.py'),'--prediction','prediction.npz','--report','nested/report.json'];r=subprocess.run(cmd,cwd=tmp_path,capture_output=True,text=True);assert r.returncode==0,r.stderr
    assert json.loads((tmp_path/'nested/report.json').read_text())['passed']
    r=subprocess.run(cmd[:-1]+[str(tmp_path)],cwd=tmp_path,capture_output=True,text=True);assert r.returncode==2 and 'Report-write' in r.stderr
