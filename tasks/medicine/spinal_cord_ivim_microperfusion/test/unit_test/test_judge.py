from pathlib import Path
import json,subprocess,sys
import numpy as np
import pytest
from evaluation import judge
ROOT=Path(__file__).resolve().parents[2]
def arrays():
    with np.load(ROOT/'evaluation/baseline_reference.npz') as z:return {k:z[k] for k in z.files}

@pytest.mark.parametrize('kind',['missing','key','shape','integer','complex','nan','inf','object','outside','range'])
def test_invalid(kind,tmp_path):
    p=tmp_path/'bad.npz';a=arrays()
    if kind=='key':del a['diffusion']
    elif kind=='shape':a['diffusion']=a['diffusion'][:,:,:-1]
    elif kind in ('integer','complex','object'):a['diffusion']=a['diffusion'].astype({'integer':int,'complex':complex,'object':object}[kind])
    elif kind in ('nan','inf'):a['diffusion'].flat[0]=float(kind)
    elif kind=='outside':a['diffusion'][0,0,0,0]=.1
    elif kind=='range':a['fraction'][a['S0']>0]=1.1
    if kind!='missing':np.savez_compressed(p,**a)
    r,c=judge.evaluate(p);assert c==2 and r['metrics']=={} and r['errors']

def test_poor_and_known_metric(tmp_path):
    a=arrays();a['S0'][a['S0']>0]+=10;p=tmp_path/'poor.npz';np.savez_compressed(p,**a)
    r,c=judge.evaluate(p);assert c==1 and r['metrics']['S0_rmse']==pytest.approx(10)

def test_metric_weights():
    r={'S0':np.array([10.,20.]),'diffusion':np.array([.001,.002]),'fraction':np.array([0.,.1]),'pseudodiffusion':np.array([.01,.02])};p={k:v.copy() for k,v in r.items()};p['pseudodiffusion'][0]=.09
    b=np.array([0.,10.]);S0,D,f,ds=[r[k][:,None] for k in judge.KEYS];obs=S0*np.exp(-b*D)*(f*np.exp(-b*ds)+1-f)
    m=judge.metrics(p,r,b,obs);assert m['Dstar_weighted_rmse']==0 and m['fDstar_rmse']==0 and m['observation_relative_l2']==0
    p['pseudodiffusion'][1]+=.002;m=judge.metrics(p,r,b,obs)
    assert m['Dstar_weighted_rmse']==pytest.approx(.002)
    assert m['fDstar_rmse']==pytest.approx(.0002/np.sqrt(2))

def test_cli_and_report_error(tmp_path):
    p=ROOT/'evaluation/baseline_reference.npz';report=tmp_path/'new/report.json'
    cmd=[sys.executable,str(ROOT/'evaluation/judge.py'),'--prediction',str(p),'--report',str(report)]
    result=subprocess.run(cmd,cwd=tmp_path,capture_output=True);assert result.returncode==0 and json.loads(report.read_text())['passed']
    blocker=tmp_path/'blocker';blocker.write_text('x');cmd[-1]=str(blocker/'r.json');assert subprocess.run(cmd,capture_output=True).returncode==2

def test_unavailable_assets(tmp_path,monkeypatch):
    monkeypatch.setattr(judge,'ROOT',tmp_path);r,c=judge.evaluate(ROOT/'evaluation/baseline_reference.npz');assert c==2 and not r['metrics']

def test_bad_boundary(monkeypatch):
    native=judge.json.loads
    def bad(s):
        q=native(s);q['D_rmse_boundary']=-1;return q
    monkeypatch.setattr(judge.json,'loads',bad);r,c=judge.evaluate(ROOT/'evaluation/baseline_reference.npz');assert c==2 and not r['metrics']
