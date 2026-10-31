from pathlib import Path
import importlib.util
import json
import subprocess
import sys
import numpy as np
import pytest
from src.visualization import metrics, observation_metrics
from src.preprocessing import load_data, SPACING

TASK = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('qsm_independent_judge', TASK/'evaluation/judge.py')
judge = importlib.util.module_from_spec(spec)
spec.loader.exec_module(judge)


@pytest.fixture(scope='module')
def reference():
    with np.load(TASK/'evaluation/baseline_reference.npz', allow_pickle=False) as z:
        return {k:z[k] for k in z.files}


def test_metric_definitions_independent_known_case():
    a = np.array([0., 1., 3.]); b = np.array([1., 2., 3.]); mask = np.ones(3, bool)
    expected_ncc = 11/np.sqrt(10*14)
    expected_nrmse = np.sqrt(2/3)/2
    for result in [judge.score(a, b, mask), metrics(a, b, mask)]:
        assert abs(result['ncc']-expected_ncc) < 1e-14
        assert abs(result['nrmse']-expected_nrmse) < 1e-14
    assert judge.score(a*0, b, mask)['ncc'] == 0
    with pytest.raises(ValueError): judge.score(a, b*0, mask)


def test_reference_pass_and_independent_pde_agreement(tmp_path, reference):
    path = tmp_path/'reference.npz'; np.savez(path, **reference)
    result, code = judge.evaluate(path)
    assert code == 0 and result['passed']
    obs = load_data(TASK/'data/raw_data.npz')
    expected = observation_metrics(reference['reconstruction'][0], reference['residual_phase'][0], obs['phase'], obs['mask'], SPACING)
    for key, value in expected.items():
        assert abs(result['metrics'][key]-value) < 1e-13


@pytest.mark.parametrize('kind', ['zero', 'wrong_sign', 'zero_chi_keep_eta', 'zero_chi_eta_equals_phase'])
def test_scientifically_degenerate_predictions_fail(tmp_path, reference, kind):
    arrays = {k:v.copy() for k,v in reference.items()}
    if kind == 'zero':
        arrays = {k:v*0 for k,v in arrays.items()}
    elif kind == 'wrong_sign':
        arrays = {k:-v for k,v in arrays.items()}
    else:
        arrays['reconstruction'][:] = 0
        if kind.endswith('equals_phase'):
            obs = load_data(TASK/'data/raw_data.npz')
            eta = obs['phase']*obs['mask']
            eta[obs['mask']] -= eta[obs['mask']].mean()
            arrays['residual_phase'] = eta[None]
    path=tmp_path/'bad.npz'; np.savez(path, **arrays)
    result, code = judge.evaluate(path)
    assert code == 1 and result['status'] == 'fail' and not result['passed']
    if kind == 'zero': assert result['metrics']['pde_relative_residual'] == pytest.approx(1.)
    if kind.endswith('equals_phase'):
        assert result['metrics']['eta_rms_rad'] > result['thresholds']['eta_rms_rad']['boundary']


@pytest.mark.parametrize('kind', ['missing_key', 'shape', 'float32', 'complex', 'nan', 'inf', 'outside', 'gauge', 'w_shape', 'w_float32', 'w_nan', 'w_outside'])
def test_invalid_prediction_reports_empty_metrics(tmp_path, reference, kind):
    arrays = {k:v.copy() for k,v in reference.items()}
    if kind == 'missing_key': arrays.pop('residual_phase')
    elif kind == 'shape': arrays['reconstruction'] = arrays['reconstruction'][:, :2]
    elif kind == 'float32': arrays['reconstruction'] = arrays['reconstruction'].astype(np.float32)
    elif kind == 'complex': arrays['reconstruction'] = arrays['reconstruction'].astype(np.complex128)
    elif kind in ['nan', 'inf']: arrays['residual_phase'][0, 80, 80, 80] = np.nan if kind == 'nan' else np.inf
    elif kind == 'outside': arrays['reconstruction'][0, 0, 0, 0] = 1
    elif kind=='w_shape': arrays['auxiliary_w']=arrays['auxiliary_w'][:,:2]
    elif kind=='w_float32': arrays['auxiliary_w']=arrays['auxiliary_w'].astype(np.float32)
    elif kind=='w_nan': arrays['auxiliary_w'].flat[0]=np.nan
    elif kind=='w_outside': arrays['auxiliary_w'].flat[0]=1
    else:
        obs=load_data(TASK/'data/raw_data.npz');arrays['reconstruction'][0][obs['mask']] += .1
    path=tmp_path/'invalid.npz'; np.savez(path, **arrays)
    result, code = judge.evaluate(path)
    assert code == 2 and result['status'] == 'error' and result['metrics'] == {} and result['errors']


def test_missing_input_and_evaluator_failure(tmp_path):
    for kwargs in [{}, {'asset_dir':tmp_path}]:
        result, code = judge.evaluate(tmp_path/'absent.npz', **kwargs)
        assert code == 2 and result['metrics'] == {} and result['errors']


def test_finite_overflow_returns_valid_error_json(tmp_path, reference):
    obs = load_data(TASK/'data/raw_data.npz')
    arrays = {key: value.copy() for key, value in reference.items()}
    indices = np.flatnonzero(obs['mask'])[:2]
    arrays['reconstruction'][:] = 0
    arrays['reconstruction'].ravel()[indices] = [1e200, -1e200]
    path = tmp_path/'finite_overflow.npz'
    np.savez(path, **arrays)
    report = tmp_path/'overflow_report.json'
    run = subprocess.run([sys.executable, str(TASK/'evaluation/judge.py'),
        '--prediction', str(path), '--report', str(report)],
        cwd=tmp_path, capture_output=True, text=True)
    assert run.returncode == 2
    result = json.loads(report.read_text(encoding='utf-8'))
    assert result['status'] == 'error' and result['metrics'] == {}
    assert any('Nonfinite computed metrics' in message for message in result['errors'])


def test_standalone_cli_foreign_working_directory_and_report_failure(tmp_path, reference):
    path=tmp_path/'prediction.npz';np.savez(path,**reference)
    report=tmp_path/'nested/report.json'
    run=subprocess.run([sys.executable,str(TASK/'evaluation/judge.py'),'--prediction',str(path),'--report',str(report)],cwd=tmp_path,capture_output=True,text=True)
    assert run.returncode == 0, run.stderr
    assert json.loads(report.read_text())['status'] == 'pass'
    blocked=tmp_path/'block';blocked.write_text('regular file')
    run=subprocess.run([sys.executable,str(TASK/'evaluation/judge.py'),'--prediction',str(path),'--report',str(blocked/'report.json')],cwd=tmp_path,capture_output=True,text=True)
    assert run.returncode == 2 and 'Cannot write evaluation report' in run.stderr
