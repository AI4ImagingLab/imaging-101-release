from pathlib import Path
import json,importlib.util,subprocess,sys
import numpy as np
import pytest
from src.preprocessing import load_data,prepare
from src.visualization import compare
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('independent_judge',ROOT/'evaluation/judge.py');judge=importlib.util.module_from_spec(spec);spec.loader.exec_module(judge)
@pytest.fixture
def reference():
    with np.load(ROOT/'evaluation/baseline_reference.npz',allow_pickle=False) as z:return {k:z[k] for k in z.files}
def save(path,data):np.savez(path,**data);return path
def test_data_precision_and_native_shape():
    s,a=load_data(ROOT/'data/raw_data.npz');assert s.shape==(2,32,32,25) and a.dtype==np.float16
    assert np.all(a==np.float16(.26171875));assert np.any(s<0);p,f=prepare(s,a);assert p.dtype==np.float64 and f.dtype==np.float64
    assert p.shape==(1024,4,25);assert np.all(p[:,2:]==0)
def test_metric_analytic_example():
    r=compare(np.array([1.,2.,4.]),np.array([1.,2.,3.]));assert r['dr_nrmse']==pytest.approx(1/np.sqrt(12));assert r['ncc']==pytest.approx(17/np.sqrt(294))
@pytest.mark.parametrize('case',['missing_key','extra_key','shape','float16','integer','complex','object','nan','inf'])
def test_malformed_prediction(tmp_path,reference,case):
    a={k:v.copy() for k,v in reference.items()};k='exchange_rate'
    if case=='missing_key':del a[k]
    elif case=='extra_key':a['extra']=np.zeros(1)
    elif case=='shape':a[k]=a[k][0]
    elif case in ('float16','integer','complex','object'):a[k]=a[k].astype({'float16':np.float16,'integer':np.int32,'complex':complex,'object':object}[case])
    else:a[k].flat[0]=np.nan if case=='nan' else np.inf
    r=judge.evaluate(save(tmp_path/'bad.npz',a));assert r['status']=='error' and r['metrics']=={};json.dumps(r,allow_nan=False)
@pytest.mark.parametrize('case',['zero','constant','scaled','negative_tiny'])
def test_poor_physical_outputs_fail(tmp_path,reference,case):
    a={k:v.copy() for k,v in reference.items()}
    if case=='negative_tiny':a['exchange_rate'].flat[0]=-np.finfo(float).tiny
    else:
        for k in a:a[k]=np.zeros_like(a[k]) if case=='zero' else np.full_like(a[k],a[k].mean()) if case=='constant' else a[k]*2
    r=judge.evaluate(save(tmp_path/'poor.npz',a));assert r['status']=='fail'
@pytest.mark.parametrize('key,value',[('ncc_boundary',1.01),('ncc_boundary',0),('ncc_boundary',True),('nrmse_boundary',-1),('max_abs_over_range_boundary',float('nan')),('negative_count_boundary',1)])
def test_invalid_limits(tmp_path,reference,key,value):
    m=json.loads((ROOT/'evaluation/metrics.json').read_text());m[key]=value;(tmp_path/'metrics.json').write_text(json.dumps(m));save(tmp_path/'baseline_reference.npz',reference)
    r=judge.evaluate(save(tmp_path/'p.npz',reference),tmp_path);assert r['status']=='error' and not r['metrics']
def test_truth_priority_invalid_truth_no_fallback_and_metric_known(tmp_path,reference):
    (tmp_path/'metrics.json').write_bytes((ROOT/'evaluation/metrics.json').read_bytes());save(tmp_path/'baseline_reference.npz',reference);p=save(tmp_path/'p.npz',reference)
    truth={k:v*2 for k,v in reference.items()};save(tmp_path/'ground_truth.npz',truth);r=judge.evaluate(p,tmp_path);assert r['status']=='fail'
    expected=np.sqrt(np.mean(reference['exchange_rate']**2))/np.ptp(truth['exchange_rate']);assert r['metrics']['exchange_rate/dr_nrmse']==pytest.approx(expected)
    truth['exchange_rate'].flat[0]=np.nan;save(tmp_path/'ground_truth.npz',truth);r=judge.evaluate(p,tmp_path);assert r['status']=='error' and not r['metrics']
def test_missing_assets_and_overflow(tmp_path,reference):
    assert judge.evaluate(tmp_path/'missing.npz')['status']=='error'
    (tmp_path/'metrics.json').write_bytes((ROOT/'evaluation/metrics.json').read_bytes());p=save(tmp_path/'p.npz',reference);assert judge.evaluate(p,tmp_path)['status']=='error'
    huge={k:np.full_like(v,np.finfo(float).max) for k,v in reference.items()};r=judge.evaluate(save(tmp_path/'huge.npz',huge));assert r['status']=='error' and not r['metrics'];json.dumps(r,allow_nan=False)
def test_standalone_foreign_cwd_and_report_write_failure(tmp_path,reference):
    p=save(tmp_path/'p.npz',reference);report=tmp_path/'nested/report.json';r=subprocess.run([sys.executable,str(ROOT/'evaluation/judge.py'),'--prediction',str(p),'--report',str(report)],cwd=tmp_path,capture_output=True);assert r.returncode==0;assert json.loads(report.read_text())['passed']
    r=subprocess.run([sys.executable,str(ROOT/'evaluation/judge.py'),'--prediction',str(p),'--report',str(tmp_path)],cwd=tmp_path,capture_output=True);assert r.returncode==2 and r.stderr

@pytest.mark.parametrize('field',['exchange_rate','initial_lactate'])
def test_every_required_field_is_independently_scored(tmp_path,reference,field):
    a={k:v.copy() for k,v in reference.items()};a[field][:]=0
    r=judge.evaluate(save(tmp_path/'single_wrong.npz',a));assert r['status']=='fail';assert r['metrics'][field+'/ncc']==0

@pytest.mark.parametrize('case,exit_code,status',[('missing',2,'error'),('zero',1,'fail')])
def test_standalone_invalid_and_valid_failure_exit_codes(tmp_path,reference,case,exit_code,status):
    path=tmp_path/'input.npz'
    if case=='zero':save(path,{k:np.zeros_like(v) for k,v in reference.items()})
    report=tmp_path/'report.json';r=subprocess.run([sys.executable,str(ROOT/'evaluation/judge.py'),'--prediction',str(path),'--report',str(report)],cwd=tmp_path,capture_output=True)
    assert r.returncode==exit_code;result=json.loads(report.read_text());assert result['status']==status
    if status=='error':assert result['metrics']=={}

def test_readme_metadata_and_model_defaults_agree():
    import inspect
    from src.preprocessing import TIME_STEP,T1_P,T1_L
    from src.solvers import reconstruct
    source=(ROOT/'README.md').read_text(encoding='utf8');assert '| Temporal sample spacing h | 2 | s |' in source
    assert TIME_STEP==2 and T1_P==30 and T1_L==25
    parameters=inspect.signature(reconstruct).parameters
    assert parameters['t1p'].default==T1_P and parameters['t1l'].default==T1_L and parameters['initial_lactate'].default==1
