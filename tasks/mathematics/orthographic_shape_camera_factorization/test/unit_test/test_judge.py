from pathlib import Path
import importlib.util,json,subprocess,sys
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('independent_judge',ROOT/'evaluation/judge.py');j=importlib.util.module_from_spec(spec);spec.loader.exec_module(j)
def ref():
    with np.load(ROOT/'evaluation/baseline_reference.npz',allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def save(tmp_path,d):
    p=tmp_path/'prediction.npz';np.savez(p,**d);return p
@pytest.mark.parametrize('key',list(j.SHAPES))
@pytest.mark.parametrize('kind',['zero','gain','negative','huge'])
def test_required_outputs_individually_bad(tmp_path,key,kind):
    d=ref();a=d[key]
    if kind=='zero':a[:]=0
    elif kind=='gain':a*=1.01
    elif kind=='negative':a*=-1
    else:a[:]=1e308
    report,code=j.evaluate(save(tmp_path,d));assert code==1 and not report['passed'];json.dumps(report,allow_nan=False)
@pytest.mark.parametrize('kind',['absent','corrupt','missingkey','extrakey','shape','integer','complex','nan','inf'])
def test_invalid_prediction(tmp_path,kind):
    p=tmp_path/'p.npz';d=ref()
    if kind=='absent':pass
    elif kind=='corrupt':p.write_bytes(b'notNPZ')
    else:
        if kind=='missingkey':del d['cameras']
        elif kind=='extrakey':d['unknown']=np.ones(1)
        elif kind=='shape':d['cameras']=d['cameras'][0]
        elif kind=='integer':d['cameras']=d['cameras'].astype(int)
        elif kind=='complex':d['cameras']=d['cameras'].astype(complex)
        else:d['cameras'][0,0,0,0]=np.nan if kind=='nan' else np.inf
        np.savez(p,**d)
    report,code=j.evaluate(p);assert code==2 and report['metrics']=={} and report['errors']
def test_joint_reflection_passes_and_float32(tmp_path):
    d=ref();Q=np.array([[0.,-1.,0.],[1.,0.,0.],[0.,0.,-1.]])
    d['reconstruction']=d['reconstruction']@Q;d['cameras']=d['cameras']@Q
    p=save(tmp_path,{k:v.astype('float32') for k,v in d.items()});report,code=j.evaluate(p);assert code==0,report
def test_separate_camera_gauge_cannot_pass(tmp_path):
    d=ref();d['cameras']=d['cameras']@np.diag([-1,1,1]);report,code=j.evaluate(save(tmp_path,d));assert code==1
def test_invalid_evaluator(tmp_path):
    report,code=j.evaluate(save(tmp_path,ref()),tmp_path);assert code==2 and report['metrics']=={}
def test_cli_foreign_workdir_and_write_error(tmp_path):
    p=save(tmp_path,ref());report=tmp_path/'nested/r.json'
    r=subprocess.run([sys.executable,str(ROOT/'evaluation/judge.py'),'--prediction',str(p),'--report',str(report)],cwd=tmp_path,capture_output=True,text=True)
    assert r.returncode==0,r.stderr
    blocked=tmp_path/'file';blocked.write_text('x')
    r=subprocess.run([sys.executable,str(ROOT/'evaluation/judge.py'),'--prediction',str(p),'--report',str(blocked/'r.json')],capture_output=True,text=True);assert r.returncode==2
