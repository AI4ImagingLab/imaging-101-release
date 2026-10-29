from pathlib import Path
import importlib.util
import json
import subprocess
import sys
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('standalone_judge',ROOT/'evaluation/judge.py')
judge=importlib.util.module_from_spec(spec);spec.loader.exec_module(judge)


def reference():
    return np.load(ROOT/'evaluation/baseline_reference.npz')['reconstruction']


def test_metric_known_case():
    assert judge.score(np.array([2.,0]),np.array([1.,0]))==1


@pytest.mark.parametrize('kind',['missing','key','shape','dtype','complex','nan','inf','poor','good'])
def test_contract(tmp_path,kind):
    x=reference(); target=tmp_path/'prediction.npz'; key='reconstruction'
    if kind=='key':key='other'
    if kind=='shape':x=x[...,0]
    if kind=='dtype':x=x.astype(np.float32)
    if kind=='complex':x=x.astype(complex)
    if kind=='nan':x.flat[0]=np.nan
    if kind=='inf':x.flat[0]=np.inf
    if kind=='poor':x*=0
    if kind!='missing':np.savez(target,**{key:x})
    report,code=judge.evaluate(target)
    assert code==({'good':0,'poor':1}.get(kind,2))
    if code==2:assert report['metrics']=={} and report['errors']
    json.dumps(report,allow_nan=False)


def test_evaluator_assets_fail(tmp_path):
    report,code=judge.evaluate(ROOT/'output/reconstruction.npz',tmp_path)
    assert code==2 and report['metrics']=={}


@pytest.mark.parametrize('kind',['missing_reference','zero_reference','bad_boundary','bad_reference_shape'])
def test_evaluator_invalid_configuration(tmp_path,kind):
    (tmp_path/'metrics.json').write_text(json.dumps({'relative_l2_boundary':-1 if kind=='bad_boundary' else 1e-6}))
    x=reference()
    if kind=='zero_reference':x*=0
    if kind=='bad_reference_shape':x=x[...,0]
    if kind!='missing_reference':np.savez(tmp_path/'baseline_reference.npz',reconstruction=x)
    report,code=judge.evaluate(ROOT/'output/reconstruction.npz',tmp_path)
    assert code==2 and report['metrics']=={} and report['errors']


def test_foreign_cwd_and_report_write_failure(tmp_path):
    target=tmp_path/'nested/report.json'
    cmd=[sys.executable,str(ROOT/'evaluation/judge.py'),'--prediction',str(ROOT/'evaluation/baseline_reference.npz'),'--report',str(target)]
    assert subprocess.run(cmd,cwd=tmp_path,capture_output=True).returncode==0
    assert json.loads(target.read_text())['passed']
    blocker=tmp_path/'blocked';blocker.write_text('file')
    cmd[-1]=str(blocker/'report.json')
    assert subprocess.run(cmd,cwd=tmp_path,capture_output=True).returncode==2
