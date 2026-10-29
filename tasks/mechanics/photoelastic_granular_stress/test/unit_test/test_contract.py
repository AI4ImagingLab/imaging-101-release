from pathlib import Path
import importlib.util,json,subprocess,sys
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('judge',ROOT/'evaluation/judge.py');j=importlib.util.module_from_spec(spec);spec.loader.exec_module(j)
def reference():
    with np.load(ROOT/'evaluation/baseline_reference.npz') as f:return {k:f[k].copy() for k in f.files}
def test_known_metrics():
    r=np.array([0.,1.,2.]);assert j.agreement(r,r)==pytest.approx((1,0));assert j.agreement(np.zeros(3),r)==pytest.approx((0,np.sqrt(5/3)/2))
    from src.visualization import agreement
    assert agreement(r+1,r)==pytest.approx(j.agreement(r+1,r))
@pytest.mark.parametrize('key',['reconstruction','forces','contact_directions'])
@pytest.mark.parametrize('case',['key','shape','dtype','nan','inf','poor'])
def test_each_quantity(key,case,tmp_path):
    d=reference();x=d[key]
    if case=='key':del d[key]
    elif case=='shape':d[key]=x[0]
    elif case=='dtype':d[key]=x.astype('float32')
    elif case=='nan':x.flat[0]=np.nan
    elif case=='inf':x.flat[0]=np.inf
    else:x[:]=0
    path=tmp_path/'prediction.npz';np.savez(path,**d);report,code=j.evaluate(path)
    assert code==(1 if case=='poor' else 2) and not report['passed']
    if code==2:assert report['metrics']=={} and report['errors']
def test_missing_corrupt_and_evaluator(tmp_path):
    assert j.evaluate(tmp_path/'missing.npz')[1]==2
    p=tmp_path/'bad.npz';p.write_text('broken');assert j.evaluate(p)[1]==2
    assert j.evaluate(ROOT/'output/reconstruction.npz',tmp_path)[1]==2
    (tmp_path/'metrics.json').write_bytes((ROOT/'evaluation/metrics.json').read_bytes());d=reference();np.savez(tmp_path/'baseline_reference.npz',**d)
    d['forces']*=2;np.savez(tmp_path/'ground_truth.npz',**d);assert j.evaluate(ROOT/'output/reconstruction.npz',tmp_path)[1]==1
    d['forces']=np.zeros((2,));np.savez(tmp_path/'ground_truth.npz',**d);assert j.evaluate(ROOT/'output/reconstruction.npz',tmp_path)[1]==2
def test_cli_and_report_failure(tmp_path):
    cmd=[sys.executable,str(ROOT/'evaluation/judge.py'),'--prediction',str(ROOT/'output/reconstruction.npz'),'--report',str(tmp_path/'nested/report.json')]
    assert subprocess.run(cmd,cwd=tmp_path,capture_output=True).returncode==0
    assert json.loads((tmp_path/'nested/report.json').read_text())['passed']
    cmd[-1]=str(tmp_path);r=subprocess.run(cmd,capture_output=True);assert r.returncode==2 and b'Cannot write' in r.stderr


def test_independent_static_metric_cases(tmp_path):
    d=reference();d['forces']=np.array([[1.,0.,0.,0.]]);d['contact_directions']=np.zeros((1,4));path=tmp_path/'static.npz';np.savez(path,**d)
    report,code=j.evaluate(path);assert code==1;assert report['metrics']['net_force_fraction']==pytest.approx(1);assert report['metrics']['torque_fraction']==pytest.approx(0)
    d['forces']=np.ones((1,4));d['contact_directions']=np.full((1,4),.1);np.savez(path,**d)
    report,code=j.evaluate(path);assert code==1;assert report['metrics']['net_force_fraction']<1e-15;assert report['metrics']['torque_fraction']==pytest.approx(np.sin(.1))
