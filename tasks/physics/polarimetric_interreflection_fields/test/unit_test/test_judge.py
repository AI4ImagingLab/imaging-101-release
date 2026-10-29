from pathlib import Path
import json,os,shutil,subprocess,sys
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[2]

@pytest.fixture(scope='module')
def method_reference():
    with np.load(ROOT/'evaluation/baseline_reference.npz',allow_pickle=False) as z:return z['reconstruction']

def run(tmp_path,a,judge=None):
    prediction=tmp_path/'prediction.npz';report=tmp_path/'nested/report.json'
    if a is not None:np.savez_compressed(prediction,reconstruction=a)
    child=subprocess.run([sys.executable,str(judge or ROOT/'evaluation/judge.py'),'--prediction',str(prediction),'--report',str(report)],cwd=tmp_path,capture_output=True,text=True)
    result=json.loads(report.read_text(encoding='utf-8'))
    assert set(result)=={'status','passed','metrics','thresholds','errors'}
    return child.returncode,result

def copied_evaluator(tmp_path):
    root=tmp_path/'independent_fixture';e=root/'evaluation';shutil.copytree(ROOT/'evaluation',e)
    (root/'data').mkdir();os.link(ROOT/'data/raw_data.npz',root/'data/raw_data.npz')
    return e

@pytest.mark.parametrize('precision',['float32','float64'])
def test_valid_foreign_cwd_and_both_declared_precisions(tmp_path,method_reference,precision):
    code,r=run(tmp_path,method_reference.astype(precision))
    assert code==0 and r['passed'] and r['status']=='pass'
    assert len(r['metrics'])==6 and r['metrics']['observation_relative_l2']>.15

@pytest.mark.parametrize('kind',['zero','scale','channel_permutation','lose_sum_imaginary','lose_difference_imaginary','absolute_value','clamp_unpolarized'])
def test_poor_fields_fail_without_alignment_or_missing_channels(tmp_path,method_reference,kind):
    a=method_reference.copy()
    if kind=='zero':a.fill(0)
    elif kind=='scale':a*=1.01
    elif kind=='channel_permutation':a=a[:,[0,3,4,1,2]]
    elif kind=='lose_sum_imaginary':a[:,2]=0
    elif kind=='lose_difference_imaginary':a[:,4]=0
    elif kind=='absolute_value':a=np.abs(a)
    else:a[:,0]=np.maximum(a[:,0],0)
    code,r=run(tmp_path,a)
    assert code==1 and not r['passed'] and r['status']=='fail' and r['metrics']

@pytest.mark.parametrize('kind',['missing','corrupt','missing_key','extra_key','shape','integer','complex','object','nan','inf'])
def test_malformed_inputs_exit2_empty_metrics(tmp_path,kind):
    a=None;path=tmp_path/'prediction.npz'
    if kind=='corrupt':path.write_bytes(b'not a zip')
    elif kind in ('missing_key','extra_key'):
        np.savez_compressed(path,**({'other':np.zeros(1)} if kind=='missing_key' else {'reconstruction':np.zeros((1,5,2048,2448)),'extra':np.ones(1)}))
    elif kind=='shape':a=np.zeros((1,5,2048,2447))
    elif kind=='integer':a=np.zeros((1,5,2048,2448),dtype=np.int8)
    elif kind=='complex':a=np.zeros((1,5,2048,2448),dtype=np.complex64)
    elif kind=='object':a=np.array([object()],dtype=object)
    elif kind in ('nan','inf'):
        a=np.zeros((1,5,2048,2448));a[0,0,0,0]=np.nan if kind=='nan' else np.inf
    code,r=run(tmp_path,a)
    assert code==2 and r['status']=='error' and not r['passed'] and r['metrics']=={} and r['errors']

@pytest.mark.parametrize('kind',['both_references_missing','reference_degenerate','bad_threshold','missing_calibration'])
def test_evaluator_asset_failures(tmp_path,method_reference,kind):
    e=copied_evaluator(tmp_path)
    if kind=='both_references_missing':(e/'baseline_reference.npz').unlink()
    elif kind=='reference_degenerate':np.savez_compressed(e/'baseline_reference.npz',reconstruction=np.zeros_like(method_reference))
    elif kind=='bad_threshold':
        p=e/'metrics.json';d=json.loads(p.read_text());d['sum_real_relative_l2_boundary']=-1;p.write_text(json.dumps(d))
    else:(e.parent/'data/raw_data.npz').unlink()
    code,r=run(tmp_path,method_reference,e/'judge.py')
    assert code==2 and r['metrics']=={} and r['errors']

@pytest.mark.parametrize('kind',['truth_priority','truth_without_baseline','malformed_truth_no_fallback'])
def test_truth_priority_is_independent_of_source_method_reference(tmp_path,method_reference,kind):
    e=copied_evaluator(tmp_path);truth=method_reference*1.0001
    np.savez_compressed(e/'ground_truth.npz',reconstruction=truth)
    if kind=='malformed_truth_no_fallback':
        (e/'ground_truth.npz').write_bytes(b'invalid truth');code,r=run(tmp_path,method_reference,e/'judge.py');assert code==2 and r['metrics']=={}
    else:
        if kind=='truth_without_baseline':(e/'baseline_reference.npz').unlink()
        code,r=run(tmp_path,truth,e/'judge.py');assert code==0 and r['passed']
        if kind=='truth_priority':
            code,r=run(tmp_path,method_reference,e/'judge.py');assert code==1 and not r['passed']

def test_report_write_failure_exit2_with_stderr(tmp_path,method_reference):
    prediction=tmp_path/'prediction.npz';np.savez_compressed(prediction,reconstruction=method_reference)
    blocked=tmp_path/'blocked';blocked.write_text('file')
    r=subprocess.run([sys.executable,str(ROOT/'evaluation/judge.py'),'--prediction',str(prediction),'--report',str(blocked/'report.json')],cwd=tmp_path,capture_output=True,text=True)
    assert r.returncode==2 and 'Cannot write report' in r.stderr
