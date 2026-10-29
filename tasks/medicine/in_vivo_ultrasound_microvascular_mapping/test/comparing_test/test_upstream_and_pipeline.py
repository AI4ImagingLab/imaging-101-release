from pathlib import Path
import numpy as np
import pytest
from src.preprocessing import load_observations,original_boundary_zero,svd_filter,temporal_filter
from src.physics_model import localize
from src.solvers import track_localizations,interpolate_tracks,render_tracks,METHODS

@pytest.fixture(scope='module')
def inputs():
    block=load_observations()[0]
    with np.load('test/fixtures/upstream_block001.npz',allow_pickle=False) as f:fixture={k:f[k] for k in f.files}
    return block,fixture

def test_independent_upstream_temporal_stages(inputs):
    block,f=inputs
    filtered=svd_filter(original_boundary_zero(block))
    assert np.linalg.norm(filtered-f['output_svd_filtered'])/np.linalg.norm(f['output_svd_filtered'])<1e-4
    out=temporal_filter(filtered)
    assert np.linalg.norm(out-f['output_filtered'])/np.linalg.norm(f['output_filtered'])<1e-4

@pytest.mark.parametrize('method',METHODS)
def test_independent_upstream_localizations(inputs,method):
    _,f=inputs;actual=localize(f['output_filtered'],method);expected=f['output_localizations_'+method]
    assert actual.shape==expected.shape
    np.testing.assert_allclose(actual[:,1:3],expected[:,1:3],atol=.001,rtol=0)
    np.testing.assert_array_equal(actual[:,3],expected[:,3])

@pytest.mark.parametrize('method',METHODS)
def test_independent_upstream_assignment_and_rendering(inputs,method):
    _,f=inputs;tracks=track_localizations(f['output_localizations_'+method])
    np.testing.assert_array_equal([len(x) for x in tracks],f['output_track_lengths_'+method])
    np.testing.assert_array_equal(np.concatenate(tracks).astype(np.float32),f['output_tracks_'+method])
    density,speed=render_tracks(interpolate_tracks(tracks))
    np.testing.assert_array_equal(density,f['output_density_'+method])
    np.testing.assert_allclose(speed,f['output_speed_'+method],rtol=2e-6,atol=2e-5)

def test_full_pipeline_submission_and_judge():
    import importlib.util
    p=Path('output/reconstruction.npz')
    if not p.exists():pytest.skip('Run python -m src.main before full pipeline test')
    spec=importlib.util.spec_from_file_location('standalone',Path('evaluation/judge.py'));m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    report,code=m.evaluate(p)
    assert code==0 and report['passed']

def test_frozen_upstream_mathematical_cases():
    from src.solvers import munkres,smooth_track
    from src.physics_model import radial_centers
    with np.load('test/fixtures/upstream_math_cases.npz',allow_pickle=False) as f:
        for i in range(5):np.testing.assert_array_equal(munkres(f[f'input_cost_{i}']),f[f'output_assignment_{i}'])
        np.testing.assert_allclose(radial_centers(f['input_patches']),f['output_centers'],rtol=0,atol=1e-13)
        np.testing.assert_allclose(smooth_track(f['input_smoothing'],4),f['output_smoothing'],rtol=0,atol=1e-14)


def test_frozen_filter_coefficients():
    from scipy.signal import butter
    with np.load('test/fixtures/upstream_block001.npz',allow_pickle=False) as f:
        b,a=butter(2,[25,249],btype='bandpass',fs=500)
        np.testing.assert_allclose(b,f['param_filter_b'],rtol=5e-14,atol=5e-15)
        np.testing.assert_allclose(a,f['param_filter_a'],rtol=5e-14,atol=5e-15)

