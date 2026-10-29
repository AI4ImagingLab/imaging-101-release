from pathlib import Path
import sys,importlib.util,numpy as np,pytest,subprocess,json,shutil
T=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('independent_sizes_judge',T/'evaluation/judge.py');judge=importlib.util.module_from_spec(spec);spec.loader.exec_module(judge)
def good():
    with np.load(T/'evaluation/baseline_reference.npz') as z:return {k:z[k].copy() for k in z}
def test_good_and_negative_quantities(tmp_path):
    for mode in ['good','zero','scaled','scrambled','background']:
        d=good()
        if mode=='zero':d['reconstruction'][:]=0
        if mode=='scaled':d['reconstruction']*=2
        if mode=='scrambled':d['reconstruction']=d['reconstruction'][:,::-1]
        if mode=='background':d['background']+=.001
        p=tmp_path/f'{mode}.npz';np.savez(p,**d);r,c=judge.evaluate(p)
        assert c==(0 if mode=='good' else 1)
        assert r['passed']==(mode=='good')
@pytest.mark.parametrize('kind',['missing','key','shape','float32','integer','nan','inf','complex','negative','pickle'])
def test_bad_contract(tmp_path,kind):
    d=good();p=tmp_path/'bad.npz'
    if kind=='missing':pass
    else:
        if kind=='key':del d['background']
        if kind=='shape':d['reconstruction']=d['reconstruction'].ravel()
        if kind=='float32':d['reconstruction']=d['reconstruction'].astype('float32')
        if kind=='integer':d['reconstruction']=np.zeros((1,500),dtype=int)
        if kind=='nan':d['reconstruction'][0,0]=np.nan
        if kind=='inf':d['background'][0,0]=np.inf
        if kind=='complex':d['background']=d['background'].astype(complex)
        if kind=='negative':d['background'][0,0]=-1
        if kind=='pickle':d['reconstruction']=np.empty((1,500),dtype=object)
        np.savez(p,**d)
    r,c=judge.evaluate(p);assert c==2 and r['metrics']=={} and r['errors']
def test_assets_caller_and_report_errors(tmp_path):
    p=tmp_path/'pred.npz';np.savez(p,**good())
    r,c=judge.evaluate(p,tmp_path);assert c==2 and not r['metrics']
    command=[sys.executable,str(T/'evaluation/judge.py'),'--prediction','pred.npz','--report','nested/report.json']
    result=subprocess.run(command,cwd=tmp_path,capture_output=True,text=True);assert result.returncode==0,result.stderr
    assert json.loads((tmp_path/'nested/report.json').read_text())['passed']
    result=subprocess.run(command[:-1]+[str(tmp_path)],cwd=tmp_path,capture_output=True,text=True);assert result.returncode==2 and 'Report-write' in result.stderr
