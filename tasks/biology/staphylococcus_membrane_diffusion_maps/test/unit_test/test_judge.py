from pathlib import Path
import importlib.util,json,subprocess,sys
import numpy as np
import pytest
from src.visualization import vector_metrics,nonlinear_mse

T=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('independent_fcs_judge',T/'evaluation/judge.py');judge=importlib.util.module_from_spec(spec);spec.loader.exec_module(judge)

@pytest.fixture
def good():
    with np.load(T/'evaluation/baseline_reference.npz',allow_pickle=False) as a:return {key:a[key] for key in judge.KEYS}

def test_known_metric_cases_and_nonlinear_runs():
    a=np.array([1.,3.]);b=np.array([2.,4.]);expected={'ncc':14/np.sqrt(200),'nrmse':.5}
    for f in (judge.vector_metrics,vector_metrics):
        assert f(a,b)==pytest.approx(expected)
        assert f(np.zeros(2),b)['ncc']==0
        with pytest.raises(ValueError):f(a,np.ones(2))
    # Residuals +1,+2,-1,-2,-3: run weights1,1,2,2,2; first prediction1.
    assert nonlinear_mse(np.array([2.,3.,0.,-1.,-2.]),np.ones(5))==pytest.approx((1+4+2+8+18)/5)

def test_exact_and_float32_roundtrip_pass(tmp_path,good):
    path=tmp_path/'prediction.npz'
    for data in (good,{k:v.astype(np.float32).astype(np.float64) for k,v in good.items()}):
        np.savez(path,**data);report,code=judge.evaluate(path)
        assert code==0 and report['status']=='pass' and len(report['metrics'])==7
        json.dumps(report,allow_nan=False)

@pytest.mark.parametrize('kind',['zeros','uniform','reverse','scale'])
def test_poor_physical_outputs_fail(tmp_path,good,kind):
    with np.load(T/'evaluation/baseline_reference.npz',allow_pickle=False) as a:mask=a['valid_mask']
    data={k:v.copy() for k,v in good.items()}
    if kind=='zeros':data={k:np.zeros_like(v) for k,v in data.items()}
    elif kind=='uniform':
        for v in data.values():v[0][mask]=np.mean(v[0][mask])
    elif kind=='reverse':
        for v in data.values():v[0][mask]=v[0][mask][::-1]
    else:data['reconstruction']*=1.2
    path=tmp_path/'bad.npz';np.savez(path,**data);report,code=judge.evaluate(path)
    assert code==1 and report['status']=='fail' and not report['passed']

@pytest.mark.parametrize('kind',['missing_key','extra_key','shape','float32','complex','nan','inf','object','background'])
def test_invalid_arrays_error(tmp_path,good,kind):
    data={k:v.copy() for k,v in good.items()};key='reconstruction'
    if kind=='missing_key':del data[key]
    elif kind=='extra_key':data['extra']=np.zeros(1)
    elif kind=='shape':data[key]=data[key][0]
    elif kind=='float32':data[key]=data[key].astype(np.float32)
    elif kind=='complex':data[key]=data[key].astype(np.complex128)
    elif kind=='object':data[key]=np.array([object()],dtype=object)
    elif kind=='background':data[key][0,0,0]=1.
    else:data[key][0,0,14]=np.nan if kind=='nan' else np.inf
    path=tmp_path/'bad.npz';np.savez(path,**data);report,code=judge.evaluate(path)
    assert code==2 and report['status']=='error' and report['metrics']=={} and report['errors']

def test_missing_broken_and_evaluator_errors(tmp_path,good):
    assert judge.evaluate(tmp_path/'absent.npz')[1]==2
    path=tmp_path/'broken.npz';path.write_bytes(b'not an NPZ');assert judge.evaluate(path)[1]==2
    np.savez(path,**good);assert judge.evaluate(path,tmp_path)[1]==2
    for name in ('baseline_reference.npz','metrics.json'):(tmp_path/name).write_bytes((T/'evaluation'/name).read_bytes())
    definition=json.loads((tmp_path/'metrics.json').read_text());definition['thresholds']['reconstruction_ncc']['boundary']=float('nan')
    (tmp_path/'metrics.json').write_text(json.dumps(definition));report,code=judge.evaluate(path,tmp_path)
    assert code==2 and report['metrics']=={}

def test_foreign_cwd_cli_and_report_write_failure(tmp_path,good):
    path=tmp_path/'good.npz';np.savez(path,**good);report=tmp_path/'nested/report.json'
    base=[sys.executable,str(T/'evaluation/judge.py'),'--prediction',str(path),'--report']
    run=subprocess.run(base+[str(report)],cwd=tmp_path,capture_output=True,text=True)
    assert run.returncode==0 and json.loads(report.read_text())['status']=='pass'
    blocker=tmp_path/'file';blocker.write_text('regular file')
    run=subprocess.run(base+[str(blocker/'report.json')],cwd=tmp_path,capture_output=True,text=True)
    assert run.returncode==2 and 'Could not write' in run.stderr

def test_actual_truth_takes_precedence_when_available(tmp_path,good):
    with np.load(T/'evaluation/baseline_reference.npz',allow_pickle=False) as a:baseline={k:a[k] for k in a.files}
    baseline['reconstruction']=baseline['reconstruction']*1.2
    np.savez(tmp_path/'baseline_reference.npz',**baseline)
    (tmp_path/'metrics.json').write_bytes((T/'evaluation/metrics.json').read_bytes())
    prediction=tmp_path/'submission.npz';np.savez(prediction,**good)
    assert judge.evaluate(prediction,tmp_path)[1]==1
    np.savez(tmp_path/'ground_truth.npz',**good)
    assert judge.evaluate(prediction,tmp_path)[1]==0
