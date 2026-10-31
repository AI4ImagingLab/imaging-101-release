from pathlib import Path
import importlib.util,sys,json,subprocess
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('judge_independent',ROOT/'evaluation/judge.py');j=importlib.util.module_from_spec(spec);spec.loader.exec_module(j)
def truth():
    with np.load(ROOT/'evaluation/ground_truth.npz',allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def save(tmp_path,d):
    p=tmp_path/'p.npz';np.savez_compressed(p,**d);return p
@pytest.mark.parametrize('key',list(j.KEYS))
@pytest.mark.parametrize('kind',['zero','negative','gain','huge'])
def test_each_required_field_bad(tmp_path,key,kind):
    d=truth()
    if kind=='zero':d[key][:]=0
    elif kind=='negative':d[key]*=-1
    elif kind=='gain':d[key]*=2
    else:d[key][:]=1e308
    report,code=j.evaluate(save(tmp_path,d));assert code==1 and not report['passed'];json.dumps(report,allow_nan=False)
@pytest.mark.parametrize('key',list(j.KEYS))
def test_lowest_floor_cannot_be_ignored(tmp_path,key):
    d=truth();d[key][:,:,0]=0;report,code=j.evaluate(save(tmp_path,d));assert code==1
@pytest.mark.parametrize('kind',['absent','corrupt','missingkey','extra','shape','integer','complex','nan','inf'])
def test_malformed(tmp_path,kind):
    p=tmp_path/'p.npz';d=truth()
    if kind=='absent':pass
    elif kind=='corrupt':p.write_bytes(b'not NPZ')
    else:
        if kind=='missingkey':del d['velocity']
        elif kind=='extra':d['unrequested']=np.ones(1)
        elif kind=='shape':d['velocity']=d['velocity'][0]
        elif kind=='integer':d['velocity']=d['velocity'].astype(int)
        elif kind=='complex':d['velocity']=d['velocity'].astype(complex)
        else:d['velocity'][0,0,0]=np.nan if kind=='nan' else np.inf
        np.savez_compressed(p,**d)
    report,code=j.evaluate(p);assert code==2 and report['metrics']=={} and report['errors']
def test_known_metrics():
    n,e=j.metrics(np.array([2.,4.]),np.array([1.,2.]));assert n==pytest.approx(1);assert e==pytest.approx(np.sqrt(2.5))
def test_copying_only_measured_channels_fails_acceleration(tmp_path):
    d=truth()
    with np.load(ROOT/'data/raw_data.npz') as z:
        acceleration=np.zeros((100001,10));t=z['reconstruction_time'][0];to=z['observation_time'][0];a=z['accelerations'][0]
        for k,floor in enumerate([0,2,4,6,8]):acceleration[:,floor]=np.interp(t,to,a[:,k])
    d['acceleration']=acceleration[None];report,code=j.evaluate(save(tmp_path,d));assert code==1
def test_integrating_without_spatial_dynamics_fails(tmp_path):
    with np.load(ROOT/'data/raw_data.npz') as z:
        t=z['reconstruction_time'][0];to=z['observation_time'][0];a=z['accelerations'][0]
    measured=np.array([np.interp(t,to,a[:,k]) for k in range(5)]).T
    weights=np.array([np.interp(np.arange(1,11),[1,3,5,7,9],np.eye(5)[k]) for k in range(5)]).T
    acc=measured@weights.T;dt=t[1]-t[0]
    vel=np.vstack((np.zeros((1,10)),np.cumsum((acc[:-1]+acc[1:])*.5*dt,axis=0)))
    dis=np.vstack((np.zeros((1,10)),np.cumsum((vel[:-1]+vel[1:])*.5*dt,axis=0)))
    report,code=j.evaluate(save(tmp_path,{'reconstruction':dis[None],'velocity':vel[None],'acceleration':acc[None]}));assert code==1
def test_float32_and_foreign_cwd_cli(tmp_path):
    p=save(tmp_path,{k:v.astype('float32') for k,v in truth().items()});report=tmp_path/'nested/report.json'
    r=subprocess.run([sys.executable,str(ROOT/'evaluation/judge.py'),'--prediction',str(p),'--report',str(report)],cwd=tmp_path,capture_output=True,text=True);assert r.returncode==0,r.stderr
    blocked=tmp_path/'blocked';blocked.write_text('x');r=subprocess.run([sys.executable,str(ROOT/'evaluation/judge.py'),'--prediction',str(p),'--report',str(blocked/'r.json')],capture_output=True,text=True);assert r.returncode==2
def test_evaluator_failure(tmp_path):
    report,code=j.evaluate(save(tmp_path,truth()),tmp_path);assert code==2 and report['metrics']=={}
