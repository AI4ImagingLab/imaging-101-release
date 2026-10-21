import importlib.util
from pathlib import Path
import shutil
import subprocess
import sys
import json
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('standalone_judge',ROOT/'evaluation/judge.py')
judge=importlib.util.module_from_spec(spec);spec.loader.exec_module(judge)


def baseline():
    with np.load(ROOT/'evaluation/baseline_reference.npz',allow_pickle=False) as z:
        return {k:z[k] for k in z.files}


def test_known_metric_case_and_zero_estimate():
    a=np.array([1.,2.,4.]);b=np.array([1.,2.,3.])
    measured=judge.compare(a,b)
    assert measured['ncc']==pytest.approx(17/np.sqrt(21*14))
    assert measured['nrmse']==pytest.approx(np.sqrt(1/3)/2)
    assert judge.compare(np.zeros(3),b)['ncc']==0
    with pytest.raises(ValueError):judge.compare(a,np.ones(3))


def test_reference_passes_and_poor_prediction_fails(tmp_path):
    assert judge.evaluate(ROOT/'evaluation/baseline_reference.npz')['status']=='pass'
    z={k:np.zeros_like(v) for k,v in baseline().items()}
    p=tmp_path/'zero.npz';np.savez(p,**z)
    r=judge.evaluate(p)
    assert r['status']=='fail' and not r['passed'] and r['metrics']


@pytest.mark.parametrize('problem',['missing_key','shape','float32','complex','nan','inf','object'])
def test_malformed_arrays(problem,tmp_path):
    z=baseline()
    if problem=='missing_key':del z['mask']
    elif problem=='shape':z['mask']=z['mask'][0]
    elif problem=='float32':z['mask']=z['mask'].astype(np.float32)
    elif problem=='complex':z['mask']=z['mask'].astype(np.complex128)
    elif problem=='object':z['mask']=z['mask'].astype(object)
    elif problem=='nan':z['mask'][0,0,0]=np.nan
    elif problem=='inf':z['mask'][0,0,0]=np.inf
    p=tmp_path/'bad.npz';np.savez(p,**z)
    r=judge.evaluate(p)
    assert r['status']=='error' and r['metrics']=={} and r['errors']


def test_missing_corrupt_files_and_evaluator_assets(tmp_path):
    assert judge.evaluate(tmp_path/'absent.npz')['status']=='error'
    p=tmp_path/'corrupt.npz';p.write_bytes(b'not an npz')
    assert judge.evaluate(p)['status']=='error'
    assert judge.evaluate(ROOT/'evaluation/baseline_reference.npz',assets=tmp_path)['status']=='error'


def test_actual_truth_takes_precedence_and_invalid_thresholds_rejected(tmp_path):
    assets=tmp_path/'task/evaluation';assets.mkdir(parents=True)
    (assets.parent/'data').mkdir()
    shutil.copy(ROOT/'data/raw_data.npz',assets.parent/'data/raw_data.npz')
    shutil.copy(ROOT/'evaluation/baseline_reference.npz',assets/'baseline_reference.npz')
    shutil.copy(ROOT/'evaluation/metrics.json',assets/'metrics.json')
    # Even an invalid actual-truth asset must take precedence; the evaluator
    # cannot silently fall back to an easier method-reference asset.
    np.savez(assets/'ground_truth.npz',mask=np.zeros((2,2)))
    r=judge.evaluate(ROOT/'evaluation/baseline_reference.npz',assets=assets)
    assert r['status']=='error' and 'Missing required key' in r['errors'][0]
    (assets/'ground_truth.npz').unlink()
    limits=json.loads((assets/'metrics.json').read_text());limits['ncc_boundary']=2
    (assets/'metrics.json').write_text(json.dumps(limits))
    r=judge.evaluate(ROOT/'evaluation/baseline_reference.npz',assets=assets)
    assert r['status']=='error' and r['metrics']=={}


def test_cli_codes_outside_task_and_unwritable_report(tmp_path):
    script=str(ROOT/'evaluation/judge.py')
    good=str(ROOT/'evaluation/baseline_reference.npz')
    report=tmp_path/'nested/report.json'
    r=subprocess.run([sys.executable,script,'--prediction',good,'--report',str(report)],cwd=tmp_path,capture_output=True)
    assert r.returncode==0
    assert json.loads(report.read_text())['passed']
    missing=tmp_path/'absent.npz'
    r=subprocess.run([sys.executable,script,'--prediction',str(missing),'--report',str(report)],capture_output=True)
    assert r.returncode==2
    bad={k:np.zeros_like(v) for k,v in baseline().items()};p=tmp_path/'poor.npz';np.savez(p,**bad)
    r=subprocess.run([sys.executable,script,'--prediction',str(p),'--report',str(report)],capture_output=True)
    assert r.returncode==1
    r=subprocess.run([sys.executable,script,'--prediction',good,'--report',str(tmp_path)],capture_output=True)
    assert r.returncode==2 and b'Cannot write report' in r.stderr


def test_public_readme_contract_and_dependencies():
    text=(ROOT/'README.md').read_text()
    for domain in ('github.com','githubusercontent.com','github.io','git.io'):
        assert domain not in text
    for key in ('observation','background','reconstruction','mask','kernel'):
        assert '| '+key+' |' in text
    assert not (ROOT/'meta_data.json').exists()
    assert '36 by 36' in text and '169, 137' in text
