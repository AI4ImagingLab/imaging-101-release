import importlib.util,json,subprocess,sys
from pathlib import Path
import numpy as np
import pytest
from numpy.testing import assert_allclose
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('judge',ROOT/'evaluation/judge.py');j=importlib.util.module_from_spec(spec);spec.loader.exec_module(j)

def baseline():
    with np.load(ROOT/'evaluation/baseline_reference.npz') as z:return {k:z[k].copy() for k in z.files}

def sandbox(tmp_path,monkeypatch):
    e=tmp_path/'evaluation';e.mkdir()
    for name in ['metrics.json','baseline_reference.npz']:(e/name).write_bytes((ROOT/'evaluation'/name).read_bytes())
    (tmp_path/'data').mkdir();(tmp_path/'data/raw_data.npz').write_bytes((ROOT/'data/raw_data.npz').read_bytes());monkeypatch.setattr(j,'HERE',e);return e

def test_known_metrics_and_angle_period(tmp_path):
    assert_allclose(j.compare(np.array([0.,2.]),np.array([0.,1.])),[1.,np.sqrt(.5)])
    a=baseline();a['parameters'][0,4]+=180
    p=tmp_path/'p.npz';np.savez(p,**a);r=j.evaluate(p);assert r['passed'];assert r['metrics']['position_angle_error']==0
    a['parameters'][0,4]=179.999999
    assert_allclose(abs((.000001-a['parameters'][0,4]+90)%180-90),.000002,atol=1e-12)

def test_independent_forward_circle():
    p=np.array([0.,0.,2.,0.,0.]);raw={'limb_points':np.array([[-2.2,0.],[1.7,0.]]),'velocity':np.tile([1.,0.],(2,1)),'timing_sigma':np.array([.1,.1]),'negative_path':np.array([[0.,3.]])}
    rms,missing,penetration=j.observation_metrics(p,raw);assert_allclose(rms,np.sqrt(6.5));assert missing==0 and penetration==0
    raw['limb_points']=np.array([[0.,4.],[0.,4.]])
    assert j.observation_metrics(p,raw)[1]==2
    raw['negative_path']=np.array([[0.,1.]])
    assert_allclose(j.observation_metrics(p,raw)[2],1.)

@pytest.mark.parametrize('mutation',['missing','shape','dtype','complex','nan','inf'])
def test_malformed_arrays(tmp_path,mutation):
    a=baseline()
    if mutation=='missing':a.pop('parameters')
    elif mutation=='shape':a['parameters']=a['parameters'][0]
    elif mutation=='dtype':a['parameters']=a['parameters'].astype('float32')
    elif mutation=='complex':a['parameters']=a['parameters'].astype('complex128')
    else:a['parameters'][0,0]=np.nan if mutation=='nan' else np.inf
    p=tmp_path/'p.npz';np.savez(p,**a)
    with pytest.raises((ValueError,KeyError)):j.evaluate(p)

@pytest.mark.parametrize('value',[None,-1.,float('nan'),float('inf'),True,'1'])
def test_strict_configuration(tmp_path,monkeypatch,value):
    e=sandbox(tmp_path,monkeypatch);c=json.loads((e/'metrics.json').read_text())
    if value is None:c.pop('radius_error_boundary')
    else:c['radius_error_boundary']=value
    (e/'metrics.json').write_text(json.dumps(c))
    with pytest.raises((ValueError,KeyError)):j.evaluate(e/'baseline_reference.npz')

def test_all_parameters_and_contour_evaluated(tmp_path):
    for index in range(5):
        a=baseline();a['parameters'][0,index]+=[10,30,30,.3,89][index]
        p=tmp_path/'p.npz';np.savez(p,**a);assert not j.evaluate(p)['passed']
    a=baseline();a['reconstruction']*=.7;np.savez(p,**a);assert not j.evaluate(p)['passed']

def test_actual_threshold_consumption_and_truth_priority(tmp_path,monkeypatch):
    e=sandbox(tmp_path,monkeypatch);a=baseline();a['parameters'][0,:4]+=[1e-5,1e-5,1e-5,1e-8];a['parameters'][0,4]+=1e-5;a['reconstruction']+=1e-5
    p=tmp_path/'p.npz';np.savez(p,**a);values=j.evaluate(p)['metrics'];c=json.loads((e/'metrics.json').read_text())
    for key in j.METRICS:
        if key in ('missing_contact_count','negative_penetration_km'):continue
        changed=dict(c);changed[key+'_boundary']=(values[key]+1)/2 if key=='contour_ncc' else values[key]/2
        (e/'metrics.json').write_text(json.dumps(changed));assert not j.evaluate(p)['passed'],key
    (e/'metrics.json').write_text(json.dumps(c));a=baseline();a['parameters'][0,0]+=10;np.savez(e/'ground_truth.npz',**a)
    assert not j.evaluate(e/'baseline_reference.npz')['passed']

def test_foreign_cli_and_report_errors(tmp_path):
    report=tmp_path/'nested/report.json';cmd=[sys.executable,str(ROOT/'evaluation/judge.py'),'--prediction',str(ROOT/'evaluation/baseline_reference.npz'),'--report',str(report)]
    assert subprocess.run(cmd,cwd=tmp_path,capture_output=True).returncode==0
    cmd[3]=str(tmp_path/'missing.npz');assert subprocess.run(cmd,cwd=tmp_path,capture_output=True).returncode==2
    result=json.loads(report.read_text());assert result['metrics']=={} and result['status']=='error'
    cmd[-1]=str(tmp_path);assert subprocess.run(cmd,cwd=tmp_path,capture_output=True).returncode==2

@pytest.mark.parametrize('boundary',[-1.01,1.01])
def test_cosine_boundary_range(tmp_path,monkeypatch,boundary):
    e=sandbox(tmp_path,monkeypatch);c=json.loads((e/'metrics.json').read_text());c['contour_ncc_boundary']=boundary;(e/'metrics.json').write_text(json.dumps(c))
    with pytest.raises(ValueError):j.evaluate(e/'baseline_reference.npz')

def test_finite_wrong_geometry_is_valid_failure(tmp_path):
    a=baseline();a['parameters'][0,2]=1.
    p=tmp_path/'p.npz';np.savez(p,**a);r=j.evaluate(p)
    assert r['status']=='fail' and r['metrics']['missing_contact_count']>0

def test_nonfinite_metric_arithmetic_is_evaluator_error(tmp_path):
    a=baseline();a['parameters'][0,2]=1e308
    p=tmp_path/'p.npz';np.savez(p,**a)
    with np.errstate(all='ignore'):
        with pytest.raises((ValueError,OverflowError,ZeroDivisionError)):j.evaluate(p)
