from pathlib import Path
import hashlib
import importlib.util
import json
import numpy as np
import pytest

from src import preprocessing as prep
from src.physics_model import dem_design
from src.solvers import clean_components, component_labels, correct_dem
from src.visualization import array_metrics

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('independent_judge', ROOT/'evaluation/judge.py')
judge = importlib.util.module_from_spec(spec)
spec.loader.exec_module(judge)


def test_readme_acquisition_constants_are_verbatim():
    text = (ROOT/'README.md').read_text(encoding='utf-8')
    for value in [prep.WAVELENGTH, prep.STARTING_RANGE, prep.RANGE_SPACING, prep.AZIMUTH_SPACING]:
        assert str(value) in text
    assert 'row 76, column 156' in text
    assert '20170910, 20180613' in text


def test_observation_archive_exact_bytes_and_raw_sample_parity():
    assert hashlib.sha256((ROOT/'data/raw_data.npz').read_bytes()).hexdigest() == '00b96aabb032e4e6b06ca062a541d29529c62ca841a3931136ff34720e3ce1fd'
    with np.load(ROOT/'data/raw_data.npz', allow_pickle=False) as data, np.load(ROOT/'test/fixtures/upstream_stages.npz', allow_pickle=False) as f:
        phase = data['phase'][0]
        measured = np.stack([phase[:, y, x] for y, x in f['coords']], axis=1)
        np.testing.assert_array_equal(measured, f['unwrapPhase'])
        components = data['components'][0]
        np.testing.assert_array_equal(data['analysis_mask'][0], np.all(components>0, axis=0))
        assert data['analysis_mask'].sum() == 159762


def test_atmospheric_referencing_uses_both_gauges_without_mutation():
    data = np.arange(3*4*5, dtype=np.float32).reshape(3, 4, 5)**2
    frozen = data.copy()
    result = prep.reference_atmosphere(data, reference=(1, 2))
    np.testing.assert_array_equal(result[:, 1, 2], 0)
    np.testing.assert_array_equal(result[0], 0)
    np.testing.assert_array_equal(data, frozen)


def test_component_connectivity_preserves_adjacent_unequal_labels():
    values = np.array([[1, 1, 2], [0, 0, 2], [1, 0, 2]])
    np.testing.assert_array_equal(component_labels(values), [[1, 1, 2], [0, 0, 2], [3, 0, 2]])


def test_component_erosion_and_small_region_rules():
    values = np.zeros((100, 100), np.uint8)
    values[5:30, 5:30] = 1
    values[60:62, 60:62] = 2
    labels = clean_components(values)
    assert labels.max() == 1 and np.count_nonzero(labels) == 625


def test_dem_design_matches_independent_upstream_and_units():
    with np.load(ROOT/'data/raw_data.npz', allow_pickle=False) as data, np.load(ROOT/'test/fixtures/upstream_stages.npz', allow_pickle=False) as f:
        design = dem_design(data['baseline'][0], data['incidence'][0], data['dates'][0])
        np.testing.assert_array_equal(design, f['G'])
    parameter = np.array([[20.], [.01], [.02], [.03], [.04], [-.05]])
    values = (design @ parameter).reshape(98, 1, 1)
    final, dem, coefficient, residual = correct_dem(values, design)
    np.testing.assert_allclose(dem, 20., atol=2e-5)
    np.testing.assert_allclose(coefficient[:, 0, 0], parameter[1:, 0], atol=1e-7)
    np.testing.assert_allclose(final[:, 0, 0], design[:, 1:]@parameter[1:, 0], atol=1e-7)
    np.testing.assert_allclose(residual, 0, atol=1e-8)


@pytest.mark.parametrize('x,y,expected', [([1., 2.], [1., 2.], (1., 0.)), ([1., 0.], [0., 2.], (0., np.sqrt(2.5)/2)), ([0., 0.], [1., 3.], (0., np.sqrt(5)/2))])
def test_notebook_and_independent_judge_known_metric_cases(x, y, expected):
    got = judge.score_pair(x, y)
    local = array_metrics(x, y)
    np.testing.assert_allclose(got, expected, atol=1e-15)
    np.testing.assert_allclose([local['ncc'], local['nrmse']], expected, atol=1e-15)


@pytest.mark.parametrize('value', [[0., 0.], [1., 1.]])
def test_degenerate_metrics_rejected(value):
    with pytest.raises(ValueError):
        judge.score_pair(value, value)
    with pytest.raises(ValueError):
        array_metrics(value, value)


@pytest.mark.parametrize('kind', ['missing_key', 'shape', 'dtype', 'complex', 'nan', 'inf'])
def test_judge_strict_array_contract(tmp_path, kind):
    shape = judge.EXPECTED['reconstruction']
    if kind == 'missing_key':
        values = {'wrong': np.zeros(1)}
    elif kind == 'shape':
        values = {'reconstruction': np.zeros((1, 2), np.float32)}
    else:
        dtype = np.float64 if kind=='dtype' else (np.complex64 if kind=='complex' else np.float32)
        array = np.zeros(shape, dtype=dtype)
        if kind in ['nan', 'inf']:
            array.flat[0] = np.nan if kind=='nan' else np.inf
        values = {'reconstruction': array}
    file = tmp_path/'bad.npz'
    np.savez_compressed(file, **values)
    with pytest.raises(ValueError):
        judge.validate_arrays(file)


def test_judge_missing_file_and_evaluator_assets(tmp_path):
    with pytest.raises(ValueError):
        judge.validate_arrays(tmp_path/'missing.npz')
    report, code = judge.evaluate(tmp_path/'missing.npz', directory=tmp_path)
    assert code==2 and report['status']=='error' and report['metrics']=={} and report['errors']


def test_loader_rejects_missing_and_shape_malformed_data(tmp_path):
    file = tmp_path/'bad.npz'
    np.savez(file, phase=np.zeros((1, 2), np.float32))
    with pytest.raises(ValueError):
        prep.load_data(file)


def test_judge_report_write_failure_returns_two(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(judge, 'evaluate', lambda path: ({'status':'error','passed':False,'metrics':{},'thresholds':{},'errors':['missing input']}, 2))
    blocking_file = tmp_path/'not_a_directory'
    blocking_file.write_text('file', encoding='utf-8')
    code = judge.main(['--prediction', 'missing.npz', '--report', str(blocking_file/'report.json')])
    assert code==2 and 'Cannot write evaluation report' in capsys.readouterr().err
