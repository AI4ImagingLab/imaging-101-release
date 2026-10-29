"""Independent original MATLAB-source captures, never cleaned self-fixtures."""
import numpy as np
import pytest

@pytest.mark.parametrize('key',['radius','labels','radius_new','initial','lower','count','radius_std','weight','anchored','reliability','contrast_limits'])
def test_original_stage(task_root,fresh,key):
    with np.load(task_root/'test/fixtures/upstream_stages.npz',allow_pickle=False) as data:
        np.testing.assert_allclose(fresh[4][key],data[key],rtol=1e-9,atol=1e-11)

def test_full_original_pipeline(task_root,fresh):
    with np.load(task_root/'evaluation/baseline_reference.npz',allow_pickle=False) as ref:
        np.testing.assert_array_equal(fresh[2][None],ref['reconstruction'])
        np.testing.assert_allclose(fresh[3][None],ref['transmission'],rtol=1e-9,atol=1e-11)

def test_normal_equations(fresh):
    s=fresh[4];x=fresh[3].ravel(order='F')
    residual=s['matrix']@x-s['rhs']
    assert np.linalg.norm(residual)/np.linalg.norm(s['rhs'])<1e-10

def test_positive_and_finite(fresh):
    t=fresh[3]
    assert np.isfinite(t).all() and t.min()>=.1 and t.max()<=1
