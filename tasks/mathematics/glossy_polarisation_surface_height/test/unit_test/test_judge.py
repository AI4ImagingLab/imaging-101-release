from pathlib import Path
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('standalone_judge',ROOT/'evaluation/judge.py')
judge=importlib.util.module_from_spec(spec);spec.loader.exec_module(judge)


def reference():
    with np.load(ROOT/'evaluation/baseline_reference.npz',allow_pickle=False) as z:return z['reconstruction']


def test_extreme_finite_prediction_is_quality_failure_with_valid_json(tmp_path):
    with np.load(ROOT/'data/raw_data.npz',allow_pickle=False) as z:mask=z['mask']
    a=np.zeros(judge.SHAPE,dtype=np.float64);a[mask]=1e308
    p=tmp_path/'huge.npz';np.savez(p,reconstruction=a)
    with np.errstate(over='raise',invalid='raise',divide='raise'):
        report,code=judge.evaluate(p)
    assert code==1 and report['status']=='fail'
    assert all(np.isfinite(v) for v in report['metrics'].values())
    json.dumps(report,allow_nan=False)


@pytest.mark.parametrize('kind',['zero','scale','flip'])
def test_quality_failures(tmp_path,kind):
    a=reference()
    if kind=='zero':a*=0
    elif kind=='scale':a*=1.01
    else:a*=-1
    p=tmp_path/'prediction.npz';np.savez_compressed(p,reconstruction=a)
    report,code=judge.evaluate(p)
    assert code==1 and report['status']=='fail' and not report['passed'] and len(report['metrics'])==2


@pytest.mark.parametrize('kind',['missing','corrupt','key','extra','shape','int','complex','object','nan','inf','background'])
def test_input_errors(tmp_path,kind):
    p=tmp_path/'prediction.npz';a=reference()
    if kind=='missing':pass
    elif kind=='corrupt':p.write_bytes(b'invalid NPZ')
    elif kind=='key':np.savez(p,other=a)
    elif kind=='extra':np.savez(p,reconstruction=a,other=np.array(1))
    else:
        if kind=='shape':a=a[0]
        elif kind=='int':a=a.astype(np.int16)
        elif kind=='complex':a=a.astype(np.complex128)
        elif kind=='object':a=a.astype(object)
        elif kind=='nan':a[0,100,100]=np.nan
        elif kind=='inf':a[0,100,100]=np.inf
        elif kind=='background':a[0,0,0]=1
        np.savez_compressed(p,reconstruction=a)
    r,c=judge.evaluate(p)
    assert c==2 and r['status']=='error' and not r['metrics'] and r['errors']


@pytest.mark.parametrize('kind',['missing_reference','degenerate','malformed_truth','threshold','calibration','truth_priority'])
def test_evaluator_assets(tmp_path,kind):
    e=tmp_path/'evaluation';shutil.copytree(ROOT/'evaluation',e)
    (tmp_path/'data').mkdir();os.link(ROOT/'data/raw_data.npz',tmp_path/'data/raw_data.npz')
    a=reference();p=tmp_path/'prediction.npz';np.savez_compressed(p,reconstruction=a)
    if kind=='missing_reference':(e/'baseline_reference.npz').unlink()
    elif kind=='degenerate':np.savez_compressed(e/'baseline_reference.npz',reconstruction=np.ones_like(a))
    elif kind=='malformed_truth':(e/'ground_truth.npz').write_bytes(b'corrupt truth')
    elif kind=='threshold':
        data=json.loads((e/'metrics.json').read_text());data['nrmse_boundary']=-1;(e/'metrics.json').write_text(json.dumps(data))
    elif kind=='calibration':(tmp_path/'data/raw_data.npz').unlink()
    else:
        truth=1.01*a;np.savez_compressed(e/'ground_truth.npz',reconstruction=truth)
        r,c=judge.evaluate(p,e);assert c==1
        np.savez_compressed(p,reconstruction=truth)
        r,c=judge.evaluate(p,e);assert c==0;return
    r,c=judge.evaluate(p,e)
    assert c==2 and not r['metrics'] and r['errors']


def test_foreign_cwd_float32_and_report_failure(tmp_path):
    p=tmp_path/'prediction.npz';np.savez_compressed(p,reconstruction=reference().astype(np.float32))
    report=tmp_path/'nested/report.json'
    run=subprocess.run([sys.executable,str(ROOT/'evaluation/judge.py'),'--prediction',str(p),'--report',str(report)],cwd=tmp_path,capture_output=True,text=True)
    assert run.returncode==0 and json.loads(report.read_text())['status']=='pass'
    blocked=tmp_path/'blocked';blocked.write_text('file')
    run=subprocess.run([sys.executable,str(ROOT/'evaluation/judge.py'),'--prediction',str(p),'--report',str(blocked/'report.json')],cwd=tmp_path,capture_output=True,text=True)
    assert run.returncode==2 and 'Cannot write' in run.stderr


def test_independent_known_cosine_and_dynamic_range(tmp_path):
    e=tmp_path/'evaluation';e.mkdir();(tmp_path/'data').mkdir()
    b=np.zeros((1,457,235));a=b.copy();mask=np.zeros(b.shape,dtype=bool)
    b[0,0,:3]=[0,1,2];a[0,0,:3]=[0,2,2];mask[0,0,:3]=True
    np.savez_compressed(e/'baseline_reference.npz',reconstruction=b)
    np.savez_compressed(tmp_path/'data/raw_data.npz',mask=mask)
    np.savez_compressed(tmp_path/'prediction.npz',reconstruction=a)
    (e/'metrics.json').write_text(json.dumps(dict(ncc_boundary=.1,nrmse_boundary=1.)))
    r,c=judge.evaluate(tmp_path/'prediction.npz',e)
    assert c==0
    np.testing.assert_allclose(r['metrics']['ncc'],6/np.sqrt(40))
    np.testing.assert_allclose(r['metrics']['nrmse'],np.sqrt(1/3)/2)
