from pathlib import Path
import numpy as np
import pytest
from src.preprocessing import load_observations
from src.physics_model import census_bytes
from src.main import reconstruct
T=Path(__file__).resolve().parents[2]

@pytest.fixture(scope='module')
def full():
    with np.load(T/'test/fixtures/original_stages.npz',allow_pickle=False) as f:fixture=dict(f)
    output,diagnostics=reconstruct(T/'data/raw_data.npz',capture_points=fixture['points_yx'])
    return output,diagnostics,fixture

def test_complete_original_census_bytes():
    left,right=load_observations(T/'data/raw_data.npz')
    with np.load(T/'test/fixtures/original_stages.npz',allow_pickle=False) as f:
        np.testing.assert_array_equal(census_bytes(left),f['census_left_bytes'])
        np.testing.assert_array_equal(census_bytes(right),f['census_right_bytes'])

def test_original_unary_and_all_eight_recursive_passes(full):
    _,diagnostics,f=full
    np.testing.assert_array_equal(diagnostics['unary_samples'],f['unary'])
    for i,row in enumerate(diagnostics['pass_samples']):np.testing.assert_allclose(row,f[f'pass_{i}'],rtol=0,atol=0.0002,equal_nan=True)
    for name in ['sum_before_correction','aggregate']:
        np.testing.assert_allclose(diagnostics[name+'_samples'],f[name],rtol=0,atol=0.0002,equal_nan=True)

def test_full_pipeline_and_independent_geometry(full):
    output,diagnostics,f=full
    with np.load(T/'notebooks/reference_outputs/upstream_method.npz',allow_pickle=False) as upstream:
        np.testing.assert_allclose(output['disparity'][0],upstream['disparity'],rtol=0,atol=1e-5)
        np.testing.assert_allclose(output['reconstruction'][0],upstream['reconstruction'],rtol=3*np.finfo(np.float32).eps,atol=0)
    expected=1038.018*176.252/(output['disparity'].astype(float)+53.271)
    assert np.max(abs(output['reconstruction']-expected)/expected)<=3*np.finfo(np.float32).eps
    assert np.isfinite(output['disparity']).all() and np.isfinite(output['reconstruction']).all()
