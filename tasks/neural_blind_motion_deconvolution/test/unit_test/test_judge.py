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
UTF8_ENV=dict(os.environ,PYTHONIOENCODING='utf-8')
spec=importlib.util.spec_from_file_location('blind_judge',ROOT/'evaluation/judge.py')
judge=importlib.util.module_from_spec(spec);spec.loader.exec_module(judge)


def baseline_arrays():
    with np.load(ROOT/'notebooks/reference_outputs/upstream.npz') as d:
        return {k:d[k].copy() for k in ('reconstruction','kernel')}


def test_valid_prediction_and_failing_identity(tmp_path):
    good=tmp_path/'good.npz';np.savez(good,**baseline_arrays())
    report,code=judge.evaluate(good);assert code==0 and report['passed']
    with np.load(ROOT/'data/raw_data.npz') as data:image=data['observation']
    k=np.zeros((1,51,51),dtype=np.float32);k[0,25,25]=1
    bad=tmp_path/'blur.npz';np.savez(bad,reconstruction=image,kernel=k)
    report,code=judge.evaluate(bad)
    assert code==1 and report['status']=='fail' and report['metrics']['forward_nrmse']<1e-10
    assert report['metrics']['nrmse']>report['thresholds']['nrmse']['boundary']


@pytest.mark.parametrize('mode',['missing','key','shape','dtype','nan','inf','complex'])
def test_malformed_predictions_have_empty_metrics(tmp_path,mode):
    path=tmp_path/'submission.npz';arrays=baseline_arrays()
    if mode=='key':arrays.pop('kernel')
    if mode=='shape':arrays['reconstruction']=arrays['reconstruction'][:,:-1]
    if mode=='dtype':arrays['kernel']=arrays['kernel'].astype(np.float64)
    if mode=='nan':arrays['kernel'][0,0,0]=np.nan
    if mode=='inf':arrays['reconstruction'][0,0,0]=np.inf
    if mode=='complex':arrays['reconstruction']=arrays['reconstruction'].astype(np.complex64)
    if mode!='missing':np.savez(path,**arrays)
    report,code=judge.evaluate(path)
    assert code==2 and report['status']=='error' and report['metrics']=={} and report['errors']


@pytest.mark.parametrize('kind',['negative','mass','mismatched'])
def test_kernel_is_actually_evaluated(tmp_path,kind):
    arrays=baseline_arrays()
    if kind=='negative':arrays['kernel'][0,0,0]=-.1
    if kind=='mass':arrays['kernel']*=2
    if kind=='mismatched':arrays['kernel']=np.roll(arrays['kernel'],15,axis=1)
    path=tmp_path/'bad_kernel.npz';np.savez(path,**arrays)
    report,code=judge.evaluate(path);assert code==1 and not report['passed']


def test_cli_is_standalone_and_cwd_independent(tmp_path):
    path=tmp_path/'prediction.npz';np.savez(path,**baseline_arrays())
    result=subprocess.run([sys.executable,str(ROOT/'evaluation/judge.py'),'--prediction',str(path),'--report','nested/report.json'],cwd=tmp_path,capture_output=True,text=True,encoding='utf-8',env=UTF8_ENV)
    assert result.returncode==0,result.stderr
    report=json.loads((tmp_path/'nested/report.json').read_text())
    assert report['status']=='pass' and not report['errors']


def test_missing_hidden_assets_and_report_write_failure(tmp_path,monkeypatch):
    path=tmp_path/'prediction.npz';np.savez(path,**baseline_arrays())
    private=tmp_path/'evaluation';private.mkdir()
    shutil.copy(ROOT/'evaluation/metrics.json',private/'metrics.json')
    monkeypatch.setattr(judge,'HERE',private)
    report,code=judge.evaluate(path)
    assert code==2 and report['metrics']=={} and report['errors']
    blocked=tmp_path/'blocked';blocked.write_text('file')
    result=subprocess.run([sys.executable,str(ROOT/'evaluation/judge.py'),'--prediction',str(path),'--report',str(blocked/'report.json')],capture_output=True,text=True,encoding='utf-8',env=UTF8_ENV)
    assert result.returncode==2 and 'Cannot write evaluation report' in result.stderr


def test_independent_metric_definition_agrees_with_notebook():
    from src.visualization import image_metrics,forward_nrmse
    a=baseline_arrays()
    with np.load(ROOT/'evaluation/ground_truth.npz') as d:truth=d['reconstruction'][0]
    with np.load(ROOT/'data/raw_data.npz') as d:obs=d['observation'][0]
    scores=judge.score_arrays(a['reconstruction'][0],a['kernel'][0],truth,obs)
    notebook=image_metrics(a['reconstruction'][0],truth)
    assert scores['ncc']==pytest.approx(notebook['ncc'],abs=1e-13)
    assert scores['nrmse']==pytest.approx(notebook['nrmse'],abs=1e-13)
    assert scores['forward_nrmse']==pytest.approx(forward_nrmse(a['reconstruction'][0],a['kernel'][0],obs),abs=1e-13)
