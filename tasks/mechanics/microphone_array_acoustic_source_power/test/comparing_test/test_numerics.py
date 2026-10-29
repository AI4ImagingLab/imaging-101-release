
from pathlib import Path
import numpy as np
from src.preprocessing import load_data
from src.solvers import reconstruct
ROOT=Path(__file__).resolve().parents[2]
def test_full_upstream_parity():
    result=reconstruct(load_data(ROOT/'data/raw_data.npz'))
    with np.load(ROOT/'evaluation/baseline_reference.npz',allow_pickle=False) as f:
        for k in f.files:
            assert result[k].shape==f[k].shape
            assert np.linalg.norm(result[k]-f[k])/np.linalg.norm(f[k])<1e-5,k

def test_cross_spectrum_geometry_and_map_stages():
    from src.preprocessing import calibrate,cross_spectral_matrix
    from src.physics_model import grid_positions,steering
    from src.solvers import solve
    d=load_data(ROOT/'data/raw_data.npz');y,valid=calibrate(d['microphone_signal'],d['calibration_factor']);c=cross_spectral_matrix(y);ref=np.load(ROOT/'test/fixtures/upstream_stages.npz')
    np.testing.assert_allclose(c,ref['output_csm'],rtol=1e-11,atol=1e-18);np.testing.assert_allclose(grid_positions().T,ref['param_grid'],atol=1e-15)
    np.testing.assert_allclose(steering(3600,d['microphone_position'][valid],grid_positions()),ref['output_steering'],rtol=3e-6,atol=1e-7)
    clean,dirty=solve(d);np.testing.assert_allclose(dirty,ref['output_das'],rtol=3e-6,atol=1e-15);np.testing.assert_array_equal(np.fft.rfftfreq(128,1/51200),ref['output_frequency']);assert np.linalg.norm(clean-ref['output_damas'])/np.linalg.norm(ref['output_damas'])<1e-5
def test_hermitian_psd_unit_psf_and_gauss_seidel():
    from src.preprocessing import cross_spectral_matrix
    from src.physics_model import point_spread,grid_positions
    from src.solvers import damas
    d=load_data(ROOT/'data/raw_data.npz');c=cross_spectral_matrix(d['microphone_signal'][:256,:3]);np.testing.assert_allclose(c,c.conj().transpose(0,2,1),atol=1e-14);assert np.linalg.eigvalsh(c).min()>-1e-14
    a=point_spread(4000,d['microphone_position'],grid_positions());np.testing.assert_allclose(np.diag(a),1,rtol=1e-12);np.testing.assert_allclose(damas(np.eye(3),np.array([1.,2.,3.]),5),[1,2,3])
