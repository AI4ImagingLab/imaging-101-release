"""Independent known metric cases and standalone submission contract checks."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location('independent_solar_judge', ROOT/'evaluation/judge.py')
judge = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(judge)


@pytest.fixture(scope='module')
def reference():
    return judge.load_arrays(ROOT/'evaluation/baseline_reference.npz')


def test_known_metrics_do_not_subtract_mean_or_fit_scale():
    reference = np.array([1., 2., 3.])
    estimate = np.array([2., 4., 6.])
    ncc, nrmse = judge.intensity_metrics(estimate, reference)
    assert ncc == pytest.approx(1.)
    assert nrmse == pytest.approx(np.sqrt(14/3)/2)
    assert judge.relative_error(estimate, reference) == pytest.approx(1.)
    assert judge.intensity_metrics(np.zeros(3), reference)[0] == 0.
    with pytest.raises(ValueError, match='Degenerate'):
        judge.intensity_metrics(reference, np.ones(3))
    with pytest.raises(ValueError, match='Zero-norm'):
        judge.relative_error(reference, np.zeros(3))


@pytest.mark.parametrize('defect', ['missing_key', 'shape', 'dtype', 'nan', 'inf', 'complex'])
def test_declared_contract_rejects_invalid_arrays(tmp_path, reference, defect):
    arrays = {key: value.copy() for key, value in reference.items()}
    key = 'wavefront_coefficients'
    if defect == 'missing_key':
        del arrays[key]
    elif defect == 'shape':
        arrays[key] = arrays[key][..., :-1]
    elif defect == 'dtype':
        arrays[key] = arrays[key].astype('float64')
    elif defect == 'complex':
        arrays[key] = arrays[key].astype('complex64')
    else:
        arrays[key].flat[0] = np.nan if defect == 'nan' else np.inf
    path = tmp_path/'invalid.npz'
    np.savez(path, **arrays)
    report, code = judge.evaluate(path)
    assert code == 2
    assert report['status'] == 'error'
    assert report['passed'] is False
    assert report['metrics'] == {}
    assert report['errors']


def test_missing_prediction_and_hidden_assets(tmp_path):
    report, code = judge.evaluate(tmp_path/'absent.npz')
    assert code == 2 and report['metrics'] == {}
    report, code = judge.evaluate(tmp_path/'absent.npz', directory=tmp_path)
    assert code == 2 and 'evaluator asset' in report['errors'][0]


def test_missing_metric_configuration_is_an_evaluator_error(tmp_path):
    (tmp_path/'baseline_reference.npz').touch()
    (tmp_path/'metrics.json').write_text(json.dumps({'thresholds': {}}))
    report, code = judge.evaluate('unused.npz', directory=tmp_path)
    assert code == 2 and report['metrics'] == {}
    assert 'metric configuration' in report['errors'][0]


def test_judge_rejects_zero_output_with_measured_finite_metrics(tmp_path, reference):
    path = tmp_path/'zero.npz'
    np.savez(path, **{key: np.zeros_like(value) for key, value in reference.items()})
    report, code = judge.evaluate(path)
    assert code == 1 and report['status'] == 'fail'
    assert report['metrics']['reconstruction_ncc_min'] == 0
    assert report['metrics']['observation_relative_error'] == pytest.approx(1.)
    assert all(np.isfinite(value) for value in report['metrics'].values())


def test_original_reference_passes_from_another_cwd(tmp_path):
    report_path = tmp_path/'new'/'report.json'
    result = subprocess.run([sys.executable, str(ROOT/'evaluation/judge.py'),
        '--prediction', str(ROOT/'evaluation/baseline_reference.npz'),
        '--report', str(report_path)], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(report_path.read_text())
    assert report['status'] == 'pass' and report['passed']
    assert report['metrics']['wavefront_psf_relative_error'] < 2e-6


def test_report_write_failure_returns_two(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(judge, 'evaluate', lambda prediction: ({'status': 'pass'}, 0))
    assert judge.main(['--prediction', 'unused', '--report', str(tmp_path)]) == 2
    assert 'Cannot write report' in capsys.readouterr().err
