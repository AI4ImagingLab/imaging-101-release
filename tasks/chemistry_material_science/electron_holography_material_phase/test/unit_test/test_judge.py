import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('judge', ROOT / 'evaluation/judge.py')
judge = importlib.util.module_from_spec(spec)
spec.loader.exec_module(judge)


def test_independent_metric_case():
    ref = np.array([1 + 1j, 1 - 1j])
    assert judge.relative_l2(2 * ref, ref) == 1.
    assert judge.relative_l2(1j * ref, ref) == pytest.approx(np.sqrt(2))
    with pytest.raises(ValueError): judge.relative_l2(ref, ref * 0)


@pytest.mark.parametrize('mode', ['missing_file', 'missing_key', 'shape', 'dtype', 'nan', 'inf', 'object', 'poor', 'pass', 'single'])
def test_prediction_contract(tmp_path, mode):
    prediction = tmp_path / 'prediction.npz'
    with np.load(ROOT / 'evaluation/baseline_reference.npz', allow_pickle=False) as f:
        x = f['reconstruction'].copy()
    if mode == 'shape': x = x[:, :-1]
    if mode == 'dtype': x = x.real
    if mode == 'nan': x[0, 0, 0] = np.nan
    if mode == 'inf': x[0, 0, 0] = np.inf
    if mode == 'object': x = np.array([object()])
    if mode == 'poor': x *= 0
    if mode == 'single': x = x.astype(np.complex64)
    if mode != 'missing_file':
        np.savez(prediction, **({'wrong': x} if mode == 'missing_key' else {'reconstruction': x}))
    result = judge.evaluate(prediction)
    wanted = 'pass' if mode in ('pass', 'single') else 'fail' if mode == 'poor' else 'error'
    assert result['status'] == wanted
    assert result['passed'] == (wanted == 'pass')
    if wanted == 'error': assert result['metrics'] == {} and result['errors']
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize('broken', ['absent_reference', 'invalid_reference', 'bad_threshold', 'missing_raw'])
def test_evaluator_failure(tmp_path, broken):
    assets = tmp_path / 'evaluation'; assets.mkdir()
    shutil.copy(ROOT / 'evaluation/metrics.json', assets)
    if broken != 'absent_reference': shutil.copy(ROOT / 'evaluation/baseline_reference.npz', assets)
    if broken == 'invalid_reference': np.savez(assets / 'baseline_reference.npz', reconstruction=np.zeros((1, 2, 2)))
    if broken == 'bad_threshold':
        config = json.loads((assets / 'metrics.json').read_text())
        config['relative_complex_l2_boundary'] = 0
        (assets / 'metrics.json').write_text(json.dumps(config))
    result = judge.evaluate(ROOT / 'evaluation/baseline_reference.npz', assets)
    assert result['status'] == 'error' and result['metrics'] == {}


@pytest.mark.parametrize('mode,code', [('good', 0), ('poor', 1), ('missing', 2), ('unwritable', 2)])
def test_cli_codes_and_path_resolution(tmp_path, mode, code):
    pred = ROOT / 'evaluation/baseline_reference.npz'
    if mode == 'poor':
        pred = tmp_path / 'poor.npz'; np.savez(pred, reconstruction=np.zeros((1, 512, 512), dtype=complex))
    if mode == 'missing': pred = tmp_path / 'missing.npz'
    report = tmp_path / 'nested/report.json'
    if mode == 'unwritable': report = tmp_path
    run = subprocess.run([sys.executable, str(ROOT / 'evaluation/judge.py'), '--prediction', str(pred), '--report', str(report)], cwd=tmp_path, capture_output=True, text=True)
    assert run.returncode == code, run.stderr
    if mode == 'unwritable': assert 'Cannot write report' in run.stderr
    else: assert json.loads(report.read_text())['status'] == {0: 'pass', 1: 'fail', 2: 'error'}[code]
