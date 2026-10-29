import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import numpy as np
import pytest

T=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('independent_judge',T/'evaluation/judge.py')
judge=importlib.util.module_from_spec(spec);spec.loader.exec_module(judge)

@pytest.fixture
def good():
    with np.load(T/'notebooks/reference_outputs/upstream_method.npz',allow_pickle=False) as data:
        return {key:data[key][None].copy() for key in ('disparity','reconstruction')}

def write(tmp_path,data):
    destination=tmp_path/'prediction.npz';np.savez_compressed(destination,**data);return destination

def test_original_standard_passes(tmp_path,good):
    report=judge.evaluate(write(tmp_path,good))
    assert report['status']=='pass' and report['passed']
    assert len(report['metrics'])==6

@pytest.mark.parametrize('problem',['missing','extra','shape','integer','complex','half','object','nan','infinity','negative_depth','bad_denominator'])
def test_strict_prediction_errors(tmp_path,good,problem):
    if problem=='missing':good.pop('disparity')
    elif problem=='extra':good['unexpected']=np.zeros(1)
    elif problem=='shape':good['disparity']=good['disparity'][0]
    elif problem in ('integer','complex','half','object'):
        good['disparity']=good['disparity'].astype({'integer':np.int32,'complex':np.complex64,'half':np.float16,'object':object}[problem])
    elif problem in ('nan','infinity'):good['disparity'][0,0,0]=np.nan if problem=='nan' else np.inf
    elif problem=='negative_depth':good['reconstruction'][0,0,0]=-1
    elif problem=='bad_denominator':good['disparity'][0,0,0]=-54
    report=judge.evaluate(write(tmp_path,good))
    assert report['status']=='error' and not report['passed'] and report['metrics']=={}

@pytest.mark.parametrize('kind',['zero','constant','shuffle','inconsistent'])
def test_poor_but_finite_predictions_fail(tmp_path,good,kind):
    if kind=='zero':good['disparity'][:]=0
    elif kind=='constant':good['disparity'][:]=np.mean(good['disparity'])
    elif kind=='shuffle':good['disparity']=np.random.RandomState(19).permutation(good['disparity'].ravel()).reshape(good['disparity'].shape)
    if kind=='inconsistent':good['reconstruction']*=1.001
    else:good['reconstruction']=(1038.018*176.252/(good['disparity'].astype(float)+53.271)).astype(np.float32)
    report=judge.evaluate(write(tmp_path,good))
    assert report['status']=='fail' and not report['passed'] and len(report['metrics'])==6

@pytest.mark.parametrize('problem',['missing_truth','bad_mask','nonfinite_truth','undefined_boundary','inconsistent_boundary'])
def test_evaluator_errors_are_distinct(tmp_path,good,monkeypatch,problem):
    assets=tmp_path/'assets';shutil.copytree(T/'evaluation',assets)
    monkeypatch.setattr(judge,'HERE',assets)
    if problem=='missing_truth':(assets/'ground_truth.npz').unlink()
    elif problem in ('bad_mask','nonfinite_truth'):
        with np.load(assets/'ground_truth.npz',allow_pickle=False) as data:ref=dict(data)
        if problem=='bad_mask':ref['valid_mask']=ref['valid_mask'].astype(np.int32)
        else:ref['disparity'][ref['valid_mask']]=np.nan
        np.savez_compressed(assets/'ground_truth.npz',**ref)
    else:
        path=assets/'metrics.json';config=json.loads(path.read_text())
        if problem=='undefined_boundary':config['thresholds']['depth_nrmse']['boundary']=None
        else:config['depth_nrmse_boundary']+=1
        path.write_text(json.dumps(config))
    report=judge.evaluate(write(tmp_path,good))
    assert report['status']=='error' and report['metrics']=={} and report['errors'][0].startswith('Evaluator error:')

def test_missing_and_invalid_archives(tmp_path):
    assert judge.evaluate(tmp_path/'missing.npz')['status']=='error'
    invalid=tmp_path/'bad.npz';invalid.write_text('not a ZIP')
    assert judge.evaluate(invalid)['status']=='error'

def test_native_cli_status_and_report_failure(tmp_path,good):
    prediction=write(tmp_path,good)
    command=[sys.executable,str(T/'evaluation/judge.py'),'--prediction',str(prediction),'--report',str(tmp_path/'nested/report.json')]
    run=subprocess.run(command,cwd=tmp_path,capture_output=True)
    assert run.returncode==0 and json.loads((tmp_path/'nested/report.json').read_text())['passed']
    good['disparity'][:]=0;good['reconstruction'][:]=1038.018*176.252/53.271
    write(tmp_path,good)
    assert subprocess.run(command,cwd=tmp_path,capture_output=True).returncode==1
    prediction.write_text('malformed')
    assert subprocess.run(command,cwd=tmp_path,capture_output=True).returncode==2
    blocker=tmp_path/'blocker';blocker.write_text('file')
    command[-1]=str(blocker/'report.json')
    run=subprocess.run(command,cwd=tmp_path,capture_output=True)
    assert run.returncode==2 and b'Report write failed' in run.stderr

def test_metric_case_matches_independent_scalar_formulas(tmp_path,good):
    report=judge.evaluate(write(tmp_path,good))
    with np.load(T/'evaluation/ground_truth.npz',allow_pickle=False) as ref:
        mask=ref['valid_mask'];x=good['disparity'][mask].astype(float);y=ref['disparity'][mask].astype(float)
        expected_cos=float(sum(x*y)/np.sqrt(sum(x*x)*sum(y*y)))
        expected_error=float(np.sqrt(sum((x-y)**2)/len(x))/(max(y)-min(y)))
    assert abs(report['metrics']['disparity_ncc']-expected_cos)<1e-12
    assert abs(report['metrics']['disparity_nrmse']-expected_error)<1e-12
