from pathlib import Path
import importlib.util,sys,subprocess,json
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
spec=importlib.util.spec_from_file_location('standalone_judge',ROOT/'evaluation/judge.py');judge=importlib.util.module_from_spec(spec);spec.loader.exec_module(judge)
from src.physics_model import density,predict
from src.preprocessing import load_data
from src.solvers import linear_reconstruct
def submit(tmp_path,r=None,s=None):
    with np.load(ROOT/'evaluation/baseline_reference.npz') as z: rr=z['reconstruction'];ss=z['intensity']
    p=tmp_path/'submission.npz';np.savez(p,reconstruction=rr if r is None else r,intensity=ss if s is None else s);return p
def test_basis_physics():
    c,p=load_data(ROOT/'data/raw_data.npz')
    assert np.allclose(p,p.conj().transpose(0,2,1))
    assert np.allclose(np.trace(p,axis1=1,axis2=2),1)
    assert np.linalg.matrix_rank(p.reshape(36,16))==16
    state=np.zeros((4,4),complex);state[0,0]=1
    a=predict(state,p).reshape(6,6)
    assert np.allclose(a[:2,:2],[[1,0],[0,0]])
    for i in (0,2,4):
        for j in (0,2,4): assert np.allclose(p.reshape(6,6,4,4)[i:i+2,j:j+2].sum(axis=(0,1)),np.eye(4))
    t=np.arange(16)*.1;r=density(t)
    assert np.linalg.eigvalsh(r).min()>-1e-14
    assert abs(np.trace(r)-1)<1e-14
    eps=1e-6;d=np.zeros((4,4),complex);d[0,3]=1j;d[3,0]=-1j
    np.testing.assert_allclose((predict(r+eps*d,p)-predict(r-eps*d,p))/(2*eps),predict(d,p),atol=1e-10)
    assert np.linalg.norm(predict(r,p)-predict(r.conj(),p))>.01
def test_good_and_independent_metrics(tmp_path):
    report,code=judge.evaluate(submit(tmp_path));assert code==0 and report['passed']
    assert report['metrics']['density_distance']==0
    assert abs(report['metrics']['observation_rmse']-.006664358465880508)<1e-12
@pytest.mark.parametrize('kind',['mixed','zero','conjugate','negative','scale'])
def test_poor(tmp_path,kind):
    with np.load(ROOT/'evaluation/baseline_reference.npz') as z:r=z['reconstruction'].copy();s=z['intensity'].copy()
    if kind=='mixed':r[0]=np.eye(4)/4
    if kind=='zero':r*=0
    if kind=='conjugate':r=r.conj()
    if kind=='negative':r[0,0,0]=-1
    if kind=='scale':s*=2
    assert judge.evaluate(submit(tmp_path,r,s))[1]==1
@pytest.mark.parametrize('kind',['missing','keys','shape','dtype','nan','inf','complexscale','object'])
def test_invalid(tmp_path,kind):
    p=tmp_path/'bad.npz';r=np.eye(4,dtype=complex)[None]/4;s=np.ones((1,1))
    if kind=='keys':np.savez(p,no=r)
    elif kind!='missing':
        if kind=='shape':r=r[0]
        if kind=='dtype':r=r.real
        if kind=='nan':r[0,0,0]=np.nan
        if kind=='inf':s[0,0]=np.inf
        if kind=='complexscale':s=s.astype(complex)
        if kind=='object':r=r.astype(object)
        np.savez(p,reconstruction=r,intensity=s)
    report,code=judge.evaluate(p);assert code==2 and report['metrics']=={} and report['errors']
def test_missing_assets(tmp_path,monkeypatch):
    p=submit(tmp_path);monkeypatch.setattr(judge,'HERE',tmp_path);assert judge.evaluate(p)[1]==2
def test_cli_outside_cwd_and_report_write(tmp_path):
    p=submit(tmp_path);dest=tmp_path/'nested'/'report.json'
    run=subprocess.run([sys.executable,str(ROOT/'evaluation/judge.py'),'--prediction',str(p),'--report',str(dest)],cwd=tmp_path,capture_output=True)
    assert run.returncode==0 and json.loads(dest.read_text())['status']=='pass'
    run=subprocess.run([sys.executable,str(ROOT/'evaluation/judge.py'),'--prediction',str(p),'--report',str(tmp_path)],capture_output=True)
    assert run.returncode==2 and b'Report write failed' in run.stderr
def test_readme_constants():
    text=(ROOT/'README.md').read_text(encoding='utf8')
    for x in ['351','702','10','36','H, V, D, A, R, L','(1, 36, 4, 4)']:assert x in text

def test_characterization_independent_cases():
    from src.visualization import state_metrics
    v=np.array([1,0,0,-1])/np.sqrt(2)
    m=state_metrics(np.outer(v,v));assert abs(m['tangle']-1)<1e-7 and abs(m['ideal_Bell_overlap']-1)<1e-12
    m=state_metrics(np.eye(4)/4);assert abs(m['entropy_bits']-2)<1e-12 and abs(m['linear_entropy']-1)<1e-12 and m['tangle']==0

def test_ground_truth_precedence_and_invalid_reference(tmp_path,monkeypatch):
    import shutil
    e=tmp_path/'evaluation';e.mkdir();(tmp_path/'data').mkdir()
    shutil.copyfile(ROOT/'data/raw_data.npz',tmp_path/'data/raw_data.npz')
    shutil.copyfile(ROOT/'evaluation/metrics.json',e/'metrics.json')
    shutil.copyfile(ROOT/'evaluation/baseline_reference.npz',e/'ground_truth.npz')
    with np.load(e/'ground_truth.npz') as z:r=z['reconstruction'];s=z['intensity']
    np.savez(e/'baseline_reference.npz',reconstruction=np.eye(4,dtype=complex)[None]/4,intensity=s)
    p=submit(tmp_path);monkeypatch.setattr(judge,'HERE',e)
    assert judge.evaluate(p)[1]==0
    bad=r.copy();bad[0,0,0]=-1
    np.savez(e/'ground_truth.npz',reconstruction=bad,intensity=s)
    report,code=judge.evaluate(p)
    assert code==2 and not report['metrics'] and 'physicality' in report['errors'][0]
