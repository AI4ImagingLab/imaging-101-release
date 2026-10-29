from pathlib import Path
import importlib.util
import subprocess
import sys
import json
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('ept_judge',ROOT/'evaluation/judge.py')
judge=importlib.util.module_from_spec(spec);spec.loader.exec_module(judge)


def reference():
    with np.load(ROOT/'evaluation/baseline_reference.npz') as a:return a['reconstruction']


@pytest.mark.parametrize('kind',['missing','bad_zip','missing_key','extra_key','shape','integer','complex','float16','object','nan','inf','outside_support'])
def test_malformed_predictions(kind,tmp_path):
    path=tmp_path/'submission.npz';data=reference().copy()
    if kind=='missing':pass
    elif kind=='bad_zip':path.write_bytes(b'not NPZ')
    elif kind=='missing_key':np.savez(path,wrong=data)
    elif kind=='extra_key':np.savez(path,reconstruction=data,extra=data)
    else:
        if kind=='shape':data=data[0]
        elif kind=='integer':data=data.astype(np.int32)
        elif kind=='complex':data=data.astype(np.complex128)
        elif kind=='float16':data=data.astype(np.float16)
        elif kind=='object':data=np.array([object()],dtype=object)
        elif kind=='nan':data.flat[0]=np.nan
        elif kind=='inf':data.flat[0]=np.inf
        elif kind=='outside_support':data.flat[0]=1
        np.savez(path,reconstruction=data)
    report=judge.evaluate(path)
    assert report['status']=='error' and not report['passed'] and report['metrics']=={} and report['errors']


@pytest.mark.parametrize('kind',['zero','negative','constant','rescaled'])
def test_valid_bad_predictions_fail(kind,tmp_path):
    data=reference().copy();mask=data!=0
    if kind=='zero':data[:]=0
    elif kind=='negative':data=-data
    elif kind=='constant':data[mask]=data[mask].mean()
    elif kind=='rescaled':data*=1.01
    path=tmp_path/'bad.npz';np.savez(path,reconstruction=data)
    report=judge.evaluate(path)
    assert report['status']=='fail' and report['metrics'] and not report['passed']


def test_native_float32_precision_and_known_metric_case(tmp_path):
    data=reference();path=tmp_path/'good.npz';np.savez(path,reconstruction=data.astype(np.float32))
    assert judge.evaluate(path)['status']=='pass'
    x=np.array([1.,2.,3.]);y=np.array([1.,2.,4.])
    assert np.dot(x,y)/np.linalg.norm(x)/np.linalg.norm(y)==pytest.approx(17/np.sqrt(14*21))


def test_independent_judge_known_numerical_case(tmp_path,monkeypatch):
    assets=tmp_path/'evaluation';assets.mkdir();data=tmp_path/'data';data.mkdir()
    (assets/'metrics.json').write_text(json.dumps(dict(ncc_boundary=.9,nrmse_boundary=.3)))
    np.savez(assets/'baseline_reference.npz',reconstruction=np.array([[1.,2.,4.]]))
    np.savez(data/'raw_data.npz',reconstruction_support=np.ones((1,3),dtype=bool))
    prediction=tmp_path/'prediction.npz';np.savez(prediction,reconstruction=np.array([[1.,2.,3.]]))
    monkeypatch.setattr(judge,'HERE',assets);monkeypatch.setattr(judge,'SHAPE',(1,3))
    report=judge.evaluate(prediction)
    assert report['metrics']['ncc']==pytest.approx(17/np.sqrt(14*21))
    assert report['metrics']['nrmse']==pytest.approx(np.sqrt(1/3)/3)


def test_evaluator_missing_asset_and_undefined_configuration(tmp_path,monkeypatch):
    monkeypatch.setattr(judge,'HERE',tmp_path)
    report=judge.evaluate(tmp_path/'does_not_exist.npz')
    assert report['status']=='error' and report['errors'][0].startswith('Evaluator error:')
    (tmp_path/'metrics.json').write_text(json.dumps(dict(ncc_boundary=None,nrmse_boundary=0)))
    assert judge.evaluate('missing')['metrics']=={}


def test_foreign_cwd_native_exit_codes_and_report_write_failure(tmp_path):
    pred=tmp_path/'pred.npz';report=tmp_path/'nested/report.json'
    np.savez(pred,reconstruction=reference())
    cmd=[sys.executable,str(ROOT/'evaluation/judge.py'),'--prediction','pred.npz','--report','nested/report.json']
    run=subprocess.run(cmd,cwd=tmp_path,capture_output=True)
    assert run.returncode==0 and json.loads(report.read_text())['status']=='pass'
    np.savez(pred,reconstruction=np.zeros_like(reference()))
    assert subprocess.run(cmd,cwd=tmp_path,capture_output=True).returncode==1
    pred.unlink()
    assert subprocess.run(cmd,cwd=tmp_path,capture_output=True).returncode==2
    pred.write_bytes(b'bad');report.parent.joinpath('blocked').write_bytes(b'file')
    cmd[-1]='nested/blocked/child.json'
    run=subprocess.run(cmd,cwd=tmp_path,capture_output=True)
    assert run.returncode==2 and b'Could not write report' in run.stderr
