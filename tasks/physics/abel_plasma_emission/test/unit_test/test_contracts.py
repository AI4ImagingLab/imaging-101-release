import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]


def judge_module():
    spec = importlib.util.spec_from_file_location('independent_judge', ROOT/'evaluation/judge.py')
    judge = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(judge)
    return judge


def test_metrics_known_values():
    judge = judge_module()
    a = np.array([0., 3., 4.])
    b = np.array([0., 0., 5.])
    actual = judge.compute_metrics(a, b)
    assert actual['ncc'] == pytest.approx(0.8)
    assert actual['nrmse'] == pytest.approx(np.sqrt(10/3)/5)
    from src.visualization import comparison_metrics
    assert comparison_metrics(a, b) == actual


@pytest.mark.parametrize('reference', [np.zeros(3), np.ones(3)])
def test_degenerate_reference_is_error(reference):
    with pytest.raises(ValueError):
        judge_module().compute_metrics(np.ones(3), reference)


@pytest.mark.parametrize('kind', ['missing', 'key', 'shape', 'dtype', 'complex', 'nan', 'inf', 'corrupt', 'truncatedzip', 'poor', 'valid'])
def test_judge_cli(kind, tmp_path):
    prediction = tmp_path/'prediction.npz'
    report = tmp_path/'nested/report.json'
    with np.load(ROOT/'notebooks/reference_outputs/upstream_reconstruction.npz') as f:
        array = f['reconstruction'].copy()
    key = 'reconstruction'
    if kind == 'key': key = 'wrong'
    if kind == 'shape': array = array[0]
    if kind == 'dtype': array = array.astype(np.int64)
    if kind == 'complex': array = array.astype(np.complex128)
    if kind == 'nan': array.flat[0] = np.nan
    if kind == 'inf': array.flat[0] = np.inf
    if kind == 'poor': array[:] = 0
    if kind == 'truncatedzip': prediction.write_bytes(b'PK\x03\x04')
    elif kind == 'corrupt': prediction.write_bytes(b'not an npz')
    elif kind != 'missing': np.savez_compressed(prediction, **{key: array})
    completed = subprocess.run([sys.executable, str(ROOT/'evaluation/judge.py'), '--prediction', str(prediction),
                                '--report', str(report)], cwd=tmp_path, capture_output=True, text=True)
    payload = json.loads(report.read_text(encoding='utf-8'))
    code = 0 if kind == 'valid' else 1 if kind == 'poor' else 2
    assert completed.returncode == code, completed.stderr
    assert payload['status'] == {0:'pass', 1:'fail', 2:'error'}[code]
    assert payload['passed'] == (code == 0)
    assert set(payload) == {'status', 'passed', 'metrics', 'thresholds', 'errors'}
    assert len(payload['metrics']) == (0 if code == 2 else 2)


@pytest.mark.parametrize('kind', ['assets', 'constant', 'threshold', 'reference_dtype', 'reference_shape'])
def test_evaluator_error(kind, tmp_path):
    judge = judge_module()
    with np.load(ROOT/'evaluation/ground_truth.npz') as data:
        truth = data['reconstruction']
    config = json.loads((ROOT/'evaluation/metrics.json').read_text())
    if kind == 'constant': truth = np.ones_like(truth)
    if kind == 'reference_dtype': truth = truth.astype(np.complex128)
    if kind == 'reference_shape': truth = truth[0]
    if kind == 'threshold': config['nrmse_boundary'] = float('nan')
    if kind != 'assets': np.savez(tmp_path/'ground_truth.npz', reconstruction=truth)
    (tmp_path/'metrics.json').write_text(json.dumps(config))
    result, code = judge.evaluate(ROOT/'notebooks/reference_outputs/upstream_reconstruction.npz', tmp_path)
    assert code == 2
    assert result['metrics'] == {}
    assert result['errors']


def test_truth_precedes_baseline(tmp_path):
    from shutil import copyfile
    for name in ['ground_truth.npz', 'metrics.json']:
        copyfile(ROOT/'evaluation'/name, tmp_path/name)
    np.savez(tmp_path/'baseline_reference.npz', reconstruction=np.ones((1,65,257)))
    result, code = judge_module().evaluate(ROOT/'notebooks/reference_outputs/upstream_reconstruction.npz', tmp_path)
    assert code == 0


def test_report_write_failure(tmp_path):
    completed = subprocess.run([sys.executable, str(ROOT/'evaluation/judge.py'), '--prediction',
                                str(ROOT/'notebooks/reference_outputs/upstream_reconstruction.npz'),
                                '--report', str(tmp_path)], capture_output=True, text=True)
    assert completed.returncode == 2
    assert 'report' in completed.stderr.lower()


@pytest.mark.parametrize('dr', [0, -1, np.nan, np.inf])
def test_invalid_spacing(dr):
    from src.solvers import inverse_transform
    with pytest.raises(ValueError): inverse_transform(np.zeros((2,5)), dr)


@pytest.mark.parametrize('value', [np.zeros((2,2,5)), np.zeros(2), np.array([0.,1.,np.nan]), np.zeros(5,dtype=complex)])
def test_invalid_transform_input(value):
    from src.solvers import inverse_transform
    with pytest.raises(ValueError): inverse_transform(value)


def test_notebook_import_preserves_backend():
    import matplotlib
    before = matplotlib.get_backend()
    import src.main
    assert matplotlib.get_backend() == before


def test_preprocessing_rejects_bad_contract(tmp_path):
    from src.preprocessing import load_projection
    np.savez(tmp_path/'bad.npz', projection=np.zeros((65,257)))
    with pytest.raises(ValueError): load_projection(tmp_path/'bad.npz')


def test_public_metadata_and_source_contract():
    from src.preprocessing import DR, DZ, NOISE_STD, SHAPE
    readme = (ROOT/'README.md').read_text(encoding='utf-8')
    assert f'| Impact/radial spacing | {DR} | mm |' in readme
    assert f'| Axial spacing | {DZ} | mm |' in readme
    assert f'| Detector noise standard deviation | {NOISE_STD} |' in readme
    with np.load(ROOT/'data/raw_data.npz', allow_pickle=False) as raw:
        assert raw.files == ['projection']
        assert raw['projection'].shape == SHAPE
        assert raw['projection'].dtype == np.float64
    for material in [readme, (ROOT/'requirements.txt').read_text()]:
        for domain in ['github.com', 'githubusercontent.com', 'github.io', 'git.io']:
            assert domain not in material.lower()
    for header in ['Background','Problem Description','Data Description','Method Hints','References']:
        assert '\n## '+header+'\n' in readme
    assert not (ROOT/'data/meta_data.json').exists()


@pytest.mark.parametrize('name', ['zeros', 'constant', 'nan', 'complex', 'shape'])
def test_visualization_metric_errors(name):
    from src.visualization import comparison_metrics
    estimate = np.array([0.,1.,2.])
    ref = estimate.copy()
    if name == 'zeros': ref[:] = 0
    if name == 'constant': ref[:] = 1
    if name == 'nan': estimate[0] = np.nan
    if name == 'complex': estimate = estimate.astype(complex)
    if name == 'shape': estimate = estimate[None]
    with pytest.raises(ValueError): comparison_metrics(estimate, ref)


def test_generator_file_contract(tmp_path):
    from src.generate_data import generate
    from src.preprocessing import load_projection
    generate(tmp_path, seed=19)
    assert load_projection(tmp_path/'data/raw_data.npz').shape == (1,65,257)
    with np.load(tmp_path/'evaluation/ground_truth.npz', allow_pickle=False) as f:
        assert f['reconstruction'].shape == (1,65,257)


def test_fallback_reference(tmp_path):
    from shutil import copyfile
    copyfile(ROOT/'evaluation/ground_truth.npz', tmp_path/'baseline_reference.npz')
    copyfile(ROOT/'evaluation/metrics.json', tmp_path/'metrics.json')
    result, code = judge_module().evaluate(ROOT/'notebooks/reference_outputs/upstream_reconstruction.npz', tmp_path)
    assert code == 0
