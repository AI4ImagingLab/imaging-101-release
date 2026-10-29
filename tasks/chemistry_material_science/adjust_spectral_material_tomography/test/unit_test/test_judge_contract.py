import importlib.util
import json
from pathlib import Path
import numpy as np
import pytest

PATH = Path(__file__).resolve().parents[2] / 'evaluation' / 'judge.py'
SPEC = importlib.util.spec_from_file_location('standalone_judge', PATH)
judge = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(judge)


@pytest.fixture
def known_case(tmp_path, monkeypatch):
    rng = np.random.default_rng(235)
    reference = {key: rng.uniform(.1, .9, shape) for key, shape in judge.SHAPES.items()}
    reference['reconstruction'] /= 5
    # Four rows and four atoms require column sums as well as row sums of one.
    reference['dictionary_weights'][0] = .6*np.eye(4)+.1*np.ones((4,4))
    basis=np.linalg.solve(reference['dictionary_weights'][0],reference['spectra'][0])
    np.savez_compressed(tmp_path / 'baseline_reference.npz', **reference,spectral_dictionary=basis[None])
    config = {}
    config['dictionary_relative_l2_boundary']=1e-6
    config['full_objective_ratio_boundary']=1.00000001
    for key in ('nonnegative_violation', 'abundance_sum_violation',
                'weight_row_sum_error', 'weight_column_sum_violation'):
        config[key + '_boundary'] = 1e-6
    config['forward_relative_l2_boundary'] = .2
    monkeypatch.setattr(judge, 'observation_consistency', lambda prediction,all_views=False: .1)
    (tmp_path / 'metrics.json').write_text(json.dumps(config))
    return tmp_path, reference


def test_independent_siddon_axis_and_diagonal_geometry():
    image = np.ones((4, 255, 255), dtype=np.float64)
    np.testing.assert_allclose(judge.ray_integrals(image, np.array([0., -1000.]),
                               np.array([0., 1000.])), 255., rtol=2e-13)
    np.testing.assert_allclose(judge.ray_integrals(image, np.array([-1000., -1000.]),
                               np.array([1000., 1000.])), 255 * np.sqrt(2), rtol=2e-13)
    image[:] = 0
    image[:, 127, 127] = 1
    np.testing.assert_allclose(judge.ray_integrals(image, np.array([0., -1000.]),
                               np.array([0., 1000.])), 1., rtol=2e-13)


def test_independent_judge_forward_against_frozen_original_line_products():
    fixture = PATH.parents[1] / 'test' / 'fixtures' / 'upstream_line_operator.npz'
    with np.load(fixture, allow_pickle=False) as reference:
        components = reference['input_maps'].shape[1]
        maps = np.stack([reference['input_maps'][:, k].reshape(255, 255, order='F')
                         for k in range(components)])
        predicted = judge.project_maps(maps, reference['param_angles'])
        actual = predicted.transpose(1, 0, 2).reshape(-1, components)
        np.testing.assert_allclose(actual, reference['output_forward'], rtol=2e-12, atol=2e-11)


def test_independent_known_metrics_and_shared_permutation(known_case):
    directory, reference = known_case
    permutation = [2, 0, 3, 1]
    candidate = {key: value[:, permutation] for key, value in reference.items()}
    path = directory / 'candidate.npz'
    np.savez_compressed(path, **candidate)
    report, code = judge.evaluate(path, directory)
    assert code == 0 and report['status'] == 'pass' and report['passed']
    assert report['metrics']['dictionary_relative_l2']<1e-15
    assert report['metrics']['full_objective_ratio']==1
    candidate['reconstruction'] = candidate['reconstruction'] * 10
    np.savez_compressed(path, **candidate)
    report, code = judge.evaluate(path, directory)
    assert code == 1 and report['status'] == 'fail'
    assert report['metrics']['abundance_sum_violation']>0


def test_zero_prediction_is_valid_failure(known_case):
    directory, reference = known_case
    path = directory / 'zero.npz'
    np.savez_compressed(path, **{key: np.zeros_like(value) for key, value in reference.items()})
    report, code = judge.evaluate(path, directory)
    assert code == 1 and not report['errors'] and report['metrics']


@pytest.mark.parametrize('damage', ['missing_file', 'missing_key', 'shape', 'dtype', 'complex', 'nan', 'inf'])
def test_malformed_submission_returns_empty_metrics(known_case, damage):
    directory, candidate = known_case
    path = directory / 'invalid.npz'
    if damage == 'missing_key':
        del candidate['spectra']
    elif damage == 'shape':
        candidate['spectra'] = candidate['spectra'][0]
    elif damage == 'dtype':
        candidate['spectra'] = candidate['spectra'].astype(np.float32)
    elif damage == 'complex':
        candidate['spectra'] = candidate['spectra'].astype(np.complex128)
    elif damage in ('nan', 'inf'):
        candidate['spectra'][0, 0, 0] = np.nan if damage == 'nan' else np.inf
    if damage != 'missing_file':
        np.savez_compressed(path, **candidate)
    report, code = judge.evaluate(path, directory)
    assert code == 2 and report['status'] == 'error'
    assert report['metrics'] == {} and report['errors']


def test_hidden_truth_precedence_and_evaluator_asset_failure(known_case):
    directory, reference = known_case
    path = directory / 'candidate.npz'
    np.savez_compressed(path, **reference)
    truth = {key: value * .5 if key != 'dictionary_weights' else value for key, value in reference.items()}
    np.savez_compressed(directory / 'ground_truth.npz', **truth)
    report, code = judge.evaluate(path, directory)
    assert code == 1 and report['status'] == 'fail'
    (directory / 'ground_truth.npz').unlink()
    (directory / 'baseline_reference.npz').unlink()
    report, code = judge.evaluate(path, directory)
    assert code == 2 and report['metrics'] == {} and report['errors']


@pytest.mark.parametrize('boundary', [None, -1, float('nan'), float('inf'), True])
def test_invalid_metric_configuration_rejected(known_case, boundary):
    directory, reference = known_case
    config = json.loads((directory / 'metrics.json').read_text())
    config['dictionary_relative_l2_boundary'] = boundary
    (directory / 'metrics.json').write_text(json.dumps(config))
    path = directory / 'candidate.npz'
    np.savez_compressed(path, **reference)
    report, code = judge.evaluate(path, directory)
    assert code == 2 and not report['metrics']


def test_cli_report_creation_and_write_failure(known_case, monkeypatch, capsys):
    directory, reference = known_case
    path = directory / 'candidate.npz'
    np.savez_compressed(path, **reference)
    original = judge.evaluate
    monkeypatch.setattr(judge, 'evaluate', lambda prediction: original(prediction, directory))
    report_path = directory / 'nested' / 'report.json'
    assert judge.main(['--prediction', str(path), '--report', str(report_path)]) == 0
    assert json.loads(report_path.read_text())['status'] == 'pass'
    assert judge.main(['--prediction', str(path), '--report', str(directory)]) == 2
    assert 'Cannot write evaluation report' in capsys.readouterr().err


def test_degenerate_reference_is_evaluator_error(known_case):
    directory, reference = known_case
    path = directory / 'candidate.npz'
    np.savez_compressed(path, **reference)
    reference['dictionary_weights'].fill(0)
    np.savez_compressed(directory / 'baseline_reference.npz', **reference)
    report, code = judge.evaluate(path, directory)
    assert code == 2 and report['metrics'] == {} and report['errors']


def test_finite_extreme_values_cannot_leak_nonfinite_json(known_case):
    directory, candidate = known_case
    candidate['reconstruction'].fill(1e308)
    path = directory / 'extreme.npz'
    np.savez_compressed(path, **candidate)
    report, code = judge.evaluate(path, directory)
    assert code == 2 and report['metrics'] == {} and report['errors']
    json.dumps(report, allow_nan=False)


def test_missing_forward_boundary_is_evaluator_error(known_case):
    directory, reference = known_case
    config = json.loads((directory / 'metrics.json').read_text())
    del config['forward_relative_l2_boundary']
    (directory / 'metrics.json').write_text(json.dumps(config))
    path = directory / 'candidate.npz'
    np.savez_compressed(path, **reference)
    report, code = judge.evaluate(path, directory)
    assert code == 2 and report['metrics'] == {} and report['errors']


def test_forward_inconsistency_fails_reference_matching_submission(known_case, monkeypatch):
    directory, reference = known_case
    path = directory / 'candidate.npz'
    np.savez_compressed(path, **reference)
    monkeypatch.setattr(judge, 'observation_consistency', lambda prediction,all_views=False: .3)
    report, code = judge.evaluate(path, directory)
    assert code == 1 and report['status'] == 'fail'
    assert report['metrics']['forward_relative_l2'] == .3
