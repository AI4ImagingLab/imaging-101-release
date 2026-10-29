from pathlib import Path
import importlib.util
import numpy as np
import pytest
import subprocess
import sys
import os

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('standalone_judge',ROOT/'evaluation/judge.py')
judge=importlib.util.module_from_spec(spec)
spec.loader.exec_module(judge)


def test_metric_known_case():
    result=judge.score(np.array([0.,2.]),np.array([0.,1.]))
    assert result['displacement_rmse_pixels']==pytest.approx(2**-.5)
    assert result['ncc']==1
    assert result['nrmse']==pytest.approx(2**-.5)


@pytest.mark.parametrize('kind',['missing','key','shape','dtype','nan','inf','complex','poor','good','corrupt','object','overflow'])
def test_submission_contract(tmp_path,kind):
    path=tmp_path/'prediction.npz'
    with np.load(ROOT/'evaluation/baseline_reference.npz',allow_pickle=False) as data:
        value=data['reconstruction'].copy()
    if kind=='key':np.savez(path,wrong=value)
    elif kind=='shape':np.savez(path,reconstruction=value[0])
    elif kind=='dtype':np.savez(path,reconstruction=value.astype(np.int32))
    elif kind=='complex':np.savez(path,reconstruction=value.astype(complex))
    elif kind in ['nan','inf']:
        value[0,0,0]=np.nan if kind=='nan' else np.inf
        np.savez(path,reconstruction=value)
    elif kind=='poor':np.savez(path,reconstruction=np.zeros_like(value))
    elif kind=='good':np.savez(path,reconstruction=value.astype(np.float32))
    elif kind=='corrupt':path.write_bytes(b'not an NPZ archive')
    elif kind=='object':np.savez(path,reconstruction=np.full((1,15,2),'object',dtype=object))
    elif kind=='overflow':np.savez(path,reconstruction=np.full((1,15,2),1e308))
    report,code=judge.evaluate(path)
    assert code==(0 if kind=='good' else 1 if kind=='poor' else 2)
    assert report['status']==('pass' if code==0 else 'fail' if code==1 else 'error')
    if code==2:assert report['metrics']=={} and report['errors']


def test_unavailable_evaluator_assets(tmp_path):
    report,code=judge.evaluate(ROOT/'evaluation/baseline_reference.npz',tmp_path)
    assert code==2 and report['metrics']=={}


def test_cli_invalid_input_and_unwritable_report(tmp_path):
    environment=dict(os.environ,PYTHONIOENCODING='utf-8')
    command=[sys.executable,str(ROOT/'evaluation/judge.py'),'--prediction','missing.npz',
             '--report','nested/error.json']
    run=subprocess.run(command,cwd=tmp_path,capture_output=True,text=True,encoding='utf-8',env=environment)
    assert run.returncode==2 and (tmp_path/'nested/error.json').exists()
    blocker=tmp_path/'file_instead_of_directory'
    blocker.write_text('report parent must be a directory')
    command=[sys.executable,str(ROOT/'evaluation/judge.py'),'--prediction',str(ROOT/'evaluation/baseline_reference.npz'),
             '--report',str(blocker/'report.json')]
    run=subprocess.run(command,cwd=tmp_path,capture_output=True,text=True,encoding='utf-8',env=environment)
    assert run.returncode==2 and 'Cannot write judge report' in run.stderr
