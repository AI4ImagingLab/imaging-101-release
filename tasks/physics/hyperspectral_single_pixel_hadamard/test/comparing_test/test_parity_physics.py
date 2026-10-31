from pathlib import Path
import numpy as np
from src.preprocessing import load_observations,differential_coefficients
from src.physics_model import walsh_matrix,forward,inverse
from src.solvers import reconstruct
ROOT=Path(__file__).resolve().parents[2]

def test_original_stages_exact():
    d=load_observations()
    with np.load(ROOT/'test/fixtures/upstream.npz',allow_pickle=False) as f:
        index=f['channel_index'];raw=d['measurements'][:,index]
        np.testing.assert_array_equal(d['pattern_id'],f['pattern_id'])
        np.testing.assert_array_equal(d['wavelength_nm'],f['wavelength_nm'])
        np.testing.assert_array_equal(raw[::2]-raw[1::2],f['differential'])
        np.testing.assert_array_equal(differential_coefficients(raw,d['pattern_id']),f['coefficients'])
        np.testing.assert_array_equal(walsh_matrix(64),f['walsh_1d'])
        np.testing.assert_array_equal(inverse(f['coefficients'].reshape(64,64,-1)),f['cube'])

def test_full_pipeline_exact_independent_original():
    output=reconstruct(load_observations())
    with np.load(ROOT/'evaluation/baseline_reference.npz',allow_pickle=False) as f:
        np.testing.assert_array_equal(output['reconstruction'],f['reconstruction'])

def test_independent_orthogonality_dc_impulse_and_adjoint():
    w=walsh_matrix(8)
    np.testing.assert_array_equal(w@w.T,8*np.eye(8))
    dc=np.ones((8,8,1));c=forward(dc)
    assert c[0,0,0]==64 and np.count_nonzero(c)==1
    impulse=np.zeros_like(dc);impulse[2,5,0]=3
    expected=3*w[:,2,None]*w[:,5][None,:]
    np.testing.assert_array_equal(forward(impulse)[...,0],expected)
    np.testing.assert_array_equal(inverse(forward(impulse)),impulse)
    rng=np.random.default_rng(501);x=rng.normal(size=(8,8,3));y=rng.normal(size=x.shape)
    np.testing.assert_allclose(np.vdot(forward(x),y),np.vdot(x,inverse(y)*64),rtol=1e-13,atol=1e-12)

def test_full_forward_observation_consistency():
    d=load_observations();x=reconstruct(d)['reconstruction'][0]
    measured=d['measurements'][::2]-d['measurements'][1::2]
    predicted=forward(x).reshape(4096,-1)[d['pattern_id'][::2]//2]
    np.testing.assert_array_equal(predicted,measured)

def test_retrospective_sampling_original_parity():
    from src.visualization import compute_experiments
    d=load_observations();fresh=reconstruct(d)['reconstruction'][0]
    result=compute_experiments(d,fresh)
    with np.load(ROOT/'test/fixtures/upstream_sampling.npz',allow_pickle=False) as f:
        np.testing.assert_array_equal(result['counts'],f['counts'])
        np.testing.assert_array_equal(result['edges_nm'],f['edges_nm'])
        np.testing.assert_array_equal(result['binned_measurements'],f['binned_measurements'])
        np.testing.assert_array_equal(result['images'][...,[0,4]],f['images'])
    assert np.all(result['nrmse'][:,-1]==0)
