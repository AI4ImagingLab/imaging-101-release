from pathlib import Path
import importlib.util
import json
import os
import subprocess
import sys
import numpy as np
import pytest
from src.preprocessing import load_data
from src.solvers import reconstruct
from src.visualization import agreement

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('judge',ROOT/'evaluation/judge.py')
judge=importlib.util.module_from_spec(spec);spec.loader.exec_module(judge)

def invoke(tmp_path,array=None,key='reconstruction',missing=False,expected=2):
    prediction=tmp_path/'prediction.npz'
    if not missing: np.savez(prediction,**{key:array})
    report=tmp_path/'nested/report.json'
    run=subprocess.run([sys.executable,str(ROOT/'evaluation/judge.py'),'--prediction',str(prediction),'--report',str(report)],cwd=tmp_path,capture_output=True,text=True,encoding='utf-8',env={**os.environ,'PYTHONIOENCODING':'utf-8'})
    assert run.returncode==expected,run.stderr
    result=json.loads(report.read_text(encoding='utf-8'))
    assert result['status']=={0:'pass',1:'fail',2:'error'}[expected]
    assert result['passed']==(expected==0)
    if expected==2: assert result['metrics']=={} and result['errors']
    return result

@pytest.mark.parametrize('case',['missing','key','shape','integer','complex','nan','inf','object','corrupt'])
def test_bad_submission(tmp_path,case):
    x=np.zeros((1,310,310),dtype=np.float64)
    if case=='missing':
        invoke(tmp_path,missing=True);return
    if case=='key':
        invoke(tmp_path,x,key='wrong');return
    if case=='shape':x=x[0]
    if case=='integer':x=x.astype(np.uint16)
    if case=='complex':x=x.astype(complex)
    if case=='object':x=x.astype(object)
    if case in ('nan','inf'):x[0,0,0]=getattr(np,case)
    if case=='corrupt':
        (tmp_path/'prediction.npz').write_bytes(b'invalid')
        report,code=judge.evaluate(tmp_path/'prediction.npz');assert code==2 and report['metrics']=={};return
    invoke(tmp_path,x)

def test_zero_and_negative_predictions_fail(tmp_path):
    invoke(tmp_path,np.zeros((1,310,310)),expected=1)
    invoke(tmp_path,-np.ones((1,310,310)),expected=1)

def test_exact_reference_passes(tmp_path):
    with np.load(ROOT/'evaluation/baseline_reference.npz') as data:x=data['reconstruction']
    invoke(tmp_path,x,expected=0)

def test_metric_known_cases():
    ref=np.array([1.,2.,3.]);estimate=np.array([3.,2.,1.])
    expected={'ncc':10/14,'nrmse':np.sqrt(8/3)/2}
    assert judge.score(estimate,ref)==pytest.approx(expected)
    assert agreement(estimate,ref)==pytest.approx(expected)
    with pytest.raises(ValueError):judge.score(np.ones(3),np.ones(3))

def test_evaluator_asset_error(tmp_path,monkeypatch):
    monkeypatch.setattr(judge,'HERE',tmp_path)
    (tmp_path/'metrics.json').write_text('{"ncc_boundary":0.9,"nrmse_boundary":0.01}')
    report,code=judge.evaluate(tmp_path/'x.npz');assert code==2 and report['metrics']=={}

def test_report_write_failure(tmp_path):
    blocked=tmp_path/'blocked';blocked.write_text('file')
    run=subprocess.run([sys.executable,str(ROOT/'evaluation/judge.py'),'--prediction','missing','--report',str(blocked/'report.json')],capture_output=True,text=True,encoding='utf-8',env={**os.environ,'PYTHONIOENCODING':'utf-8'})
    assert run.returncode==2 and 'Cannot write report' in run.stderr

def test_model_inputs_and_limits(tmp_path):
    np.savez(tmp_path/'bad.npz',observation=np.ones((1,5,5)),psf=np.ones((1,13,13)))
    with pytest.raises(ValueError):load_data(tmp_path/'bad.npz')
    with pytest.raises(ValueError):reconstruct(np.ones((8,8)),np.ones((3,3)),beta=0)
    np.testing.assert_allclose(reconstruct(np.ones((8,8)),np.ones((3,3)),pad=0),1)

def test_readme_acquisition_constants():
    text=(ROOT/'README.md').read_text(encoding='utf-8')
    for token in ['310','13','1.5','reflect','float32','Airyscan']:
        assert token in text
