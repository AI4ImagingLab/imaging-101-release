from pathlib import Path
import importlib.util
import json
import subprocess
import sys
import shutil
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[2]
JUDGE=ROOT/'evaluation/judge.py'
spec=importlib.util.spec_from_file_location('independent_judge',JUDGE)
judge=importlib.util.module_from_spec(spec);spec.loader.exec_module(judge)

def run(prediction,report,cwd=None,script=JUDGE):
    return subprocess.run([sys.executable,str(script),'--prediction',str(prediction),'--report',str(report)],cwd=cwd,capture_output=True,text=True)

def truth():
    with np.load(ROOT/'evaluation/ground_truth.npz',allow_pickle=False) as d:return d['reconstruction'].copy()

def test_metric_conventions_known_numbers():
    b=np.array([[[[1.,2.],[3.,4.]],[[2.,4.],[6.,8.]]]])
    a=b+1
    result=judge.component_metrics(a,b)
    assert result['Bx_nrmse']==1/3
    assert result['By_nrmse']==1/6
    assert result['Bx_ncc']==pytest.approx(np.dot(a[0,0].ravel(),b[0,0].ravel())/np.linalg.norm(a[0,0])/np.linalg.norm(b[0,0]))
    assert result['Bx_ncc']<1

def test_foreign_working_directory_and_actual_truth_pass(tmp_path):
    prediction=tmp_path/'good.npz';np.savez(prediction,reconstruction=truth())
    report=tmp_path/'new'/'report.json';p=run(prediction,report,cwd=tmp_path)
    assert p.returncode==0,p.stderr
    result=json.loads(report.read_text());assert result['status']=='pass' and result['passed']
    assert result['errors']==[] and all(np.isfinite(v) for v in result['metrics'].values())
    assert len(result['thresholds'])==4

@pytest.mark.parametrize('control',['zero','negative','gain','swapped'])
def test_bad_physical_results_fail(tmp_path,control):
    a=truth()
    if control=='zero':a*=0
    if control=='negative':a=-a
    if control=='gain':a*=1.2
    if control=='swapped':a=a[:,::-1]
    pred=tmp_path/'bad.npz';np.savez(pred,reconstruction=a);report=tmp_path/'r.json'
    p=run(pred,report);result=json.loads(report.read_text())
    assert p.returncode==1 and result['status']=='fail' and not result['passed']
    assert result['metrics'] and not result['errors']

@pytest.mark.parametrize('kind',['missing','missing_key','shape','integer','complex','nan','inf','object','corrupt'])
def test_malformed_submissions_return2_empty_metrics(tmp_path,kind):
    a=truth();pred=tmp_path/'bad.npz'
    if kind=='missing':pass
    elif kind=='missing_key':np.savez(pred,wrong=a)
    elif kind=='shape':np.savez(pred,reconstruction=a[0])
    elif kind=='integer':np.savez(pred,reconstruction=a.astype(np.int64))
    elif kind=='complex':np.savez(pred,reconstruction=a.astype(np.complex128))
    elif kind=='object':np.savez(pred,reconstruction=np.array([object()],dtype=object))
    elif kind=='corrupt':pred.write_bytes(b'not an archive')
    else:
        a[0,0,0,0]=np.nan if kind=='nan' else np.inf
        np.savez(pred,reconstruction=a)
    report=tmp_path/'r.json';p=run(pred,report);result=json.loads(report.read_text())
    assert p.returncode==2 and result['status']=='error' and not result['passed']
    assert result['metrics']=={} and result['errors']

def test_missing_evaluator_asset_is_error(tmp_path):
    private=tmp_path/'judge_dir';private.mkdir()
    shutil.copy2(JUDGE,private/'judge.py');shutil.copy2(ROOT/'evaluation/metrics.json',private/'metrics.json')
    pred=tmp_path/'p.npz';np.savez(pred,reconstruction=truth());report=tmp_path/'r.json'
    p=run(pred,report,script=private/'judge.py');result=json.loads(report.read_text())
    assert p.returncode==2 and result['metrics']=={} and 'Evaluator error' in result['errors'][0]

def test_invalid_evaluator_boundary_is_error(tmp_path):
    private=tmp_path/'judge_dir';private.mkdir()
    for name in ['judge.py','metrics.json','ground_truth.npz']:shutil.copy2(ROOT/'evaluation'/name,private/name)
    c=json.loads((private/'metrics.json').read_text());c['Bx_ncc_boundary']=None
    (private/'metrics.json').write_text(json.dumps(c))
    pred=tmp_path/'p.npz';np.savez(pred,reconstruction=truth());report=tmp_path/'r.json'
    p=run(pred,report,script=private/'judge.py');result=json.loads(report.read_text())
    assert p.returncode==2 and result['metrics']=={} and result['errors']

def test_report_write_failure_returns2(tmp_path):
    pred=tmp_path/'p.npz';np.savez(pred,reconstruction=truth())
    blocked=tmp_path/'blocked';blocked.write_text('file, not directory')
    p=run(pred,blocked/'r.json')
    assert p.returncode==2 and 'Cannot write evaluation report' in p.stderr

def test_standard_entrypoint_help_from_task_root():
    p=subprocess.run([sys.executable,'-m','src.main','--help'],cwd=ROOT,capture_output=True,text=True)
    assert p.returncode==0 and '--diagnostics' in p.stdout
