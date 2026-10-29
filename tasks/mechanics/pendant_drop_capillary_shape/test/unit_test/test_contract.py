from pathlib import Path
import sys,subprocess,json,importlib.util,shutil
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('drop_judge',ROOT/'evaluation/judge.py');judge=importlib.util.module_from_spec(spec);spec.loader.exec_module(judge)
def reference():
    with np.load(ROOT/'evaluation/baseline_reference.npz') as z:return {k:z[k] for k in z}
def test_distance_known_case():
    curve=np.array([[0.,0],[1,0],[2,0]]);points=np.array([[.5,3.],[1.5,4.]])
    assert abs(judge.distance_rms(points,curve)-np.sqrt(12.5))<1e-14
def test_scale_ambiguity():
    p=reference()['parameters'][0];R,B=p[2:4]
    gamma1=1000*9.81*(R*1e-5)**2/B;gamma2=1000*9.81*(R*2e-5)**2/B
    assert gamma2==4*gamma1 # identical pixel shape cannot determine metric tension
def test_readme_acquisition_contract():
    text=(ROOT/'README.md').read_text(encoding='utf8')
    for literal in ('1024','1280','293','250:760','190:720','401','float32 or float64'):assert literal in text
    with np.load(ROOT/'data/raw_data.npz') as z:
        c=z['contour'][0]
        assert np.all((c[:,0]>=250)&(c[:,0]<760)&(c[:,1]>=190)&(c[:,1]<720))
@pytest.mark.parametrize('kind',['missing','shape','complex','int','float16','nan','inf','negative','overflow'])
def test_malformed(kind,tmp_path):
    d=reference()
    if kind=='missing':d.pop('capillary_length')
    elif kind=='shape':d['parameters']=d['parameters'].ravel()
    elif kind=='complex':d['reconstruction']=d['reconstruction'].astype(complex)
    elif kind=='int':d['reconstruction']=d['reconstruction'].astype(int)
    elif kind=='float16':d['reconstruction']=d['reconstruction'].astype('float16')
    elif kind=='nan':d['parameters'][0,0]=np.nan
    elif kind=='inf':d['parameters'][0,0]=np.inf
    elif kind=='negative':d['parameters'][0,3]=-.2
    elif kind=='overflow':d['reconstruction'][:]=1e308
    path=tmp_path/'p.npz';np.savez(path,**d);r=judge.evaluate(path)
    assert r['status']=='error' and not r['metrics'] and r['errors']
@pytest.mark.parametrize('kind',['shift','bond','capillary','circle'])
def test_poor(kind,tmp_path):
    d=reference()
    if kind=='shift':d['reconstruction']+=2
    elif kind=='bond':d['parameters'][0,3]=.4
    elif kind=='capillary':d['capillary_length']*=2
    else:
        t=np.linspace(-np.pi,np.pi,401);d['reconstruction'][0]=np.array([500+190*np.sin(t),450+190*np.cos(t)]).T
    p=tmp_path/'p.npz';np.savez(p,**d);assert judge.evaluate(p)['status']=='fail'
def test_float32_and_assets(tmp_path,monkeypatch):
    p=tmp_path/'p.npz';np.savez(p,**{k:v.astype('float32') for k,v in reference().items()});assert judge.evaluate(p)['passed']
    monkeypatch.setattr(judge,'HERE',tmp_path);assert judge.evaluate(p)['status']=='error'
@pytest.mark.parametrize('fault',['negative_reference','missing_reference','bad_boundary'])
def test_hidden_errors(fault,tmp_path,monkeypatch):
    e=tmp_path/'evaluation';e.mkdir();shutil.copytree(ROOT/'data',tmp_path/'data');d=reference()
    if fault=='negative_reference':d['parameters'][0,2]=-1
    if fault!='missing_reference':np.savez(e/'baseline_reference.npz',**d)
    limits=json.loads((ROOT/'evaluation/metrics.json').read_text(encoding='utf8'))
    if fault=='bad_boundary':limits['bond_error_boundary']=-1
    (e/'metrics.json').write_text(json.dumps(limits),encoding='utf8');monkeypatch.setattr(judge,'HERE',e)
    assert judge.evaluate(ROOT/'evaluation/baseline_reference.npz')['status']=='error'
def test_cli(tmp_path):
    subprocess.run([sys.executable,'-m','src.main'],cwd=ROOT,check=True)
    p=ROOT/'output/reconstruction.npz';report=tmp_path/'nested/report.json'
    bad=reference();bad['parameters'][0,3]=.4;np.savez(tmp_path/'bad.npz',**bad)
    for path,code in [(p,0),(tmp_path/'bad.npz',1),(tmp_path/'absent.npz',2)]:
        out=subprocess.run([sys.executable,str(ROOT/'evaluation/judge.py'),'--prediction',str(path),'--report',str(report)],cwd=tmp_path)
        assert out.returncode==code
    out=subprocess.run([sys.executable,str(ROOT/'evaluation/judge.py'),'--prediction',str(p),'--report',str(tmp_path)],capture_output=True)
    assert out.returncode==2 and b'Report write failed' in out.stderr
