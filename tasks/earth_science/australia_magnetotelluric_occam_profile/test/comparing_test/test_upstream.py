from pathlib import Path
import numpy as np
from src.preprocessing import load_data
from src.physics_model import response,jacobian
from src.solvers import reconstruct

ROOT=Path(__file__).resolve().parents[2]

def test_original_observations_exact():
    with np.load(ROOT/'test/fixtures/upstream_corrected.npz') as a:
        d=load_data(ROOT/'data/raw_data.npz');p=a['input_published_observations']
        np.testing.assert_array_equal(d['period'],p[:,0])
        np.testing.assert_array_equal(d['log_apparent_resistivity'],p[:,1])
        np.testing.assert_array_equal(d['log_resistivity_std'],p[:,2])
        np.testing.assert_array_equal(d['layer_top'],a['param_layer_top'][0])
        np.testing.assert_allclose(d['impedance_phase'],p[:,3]*np.pi/180,atol=0,rtol=0)

def test_corrected_original_stages():
    with np.load(ROOT/'test/fixtures/upstream_corrected.npz') as a:
        f=a['input_frequency'][0];m=a['input_probe_model'][0];z=a['param_layer_top'][0]
        r,p=response(f,m,z);j1,j2,c=jacobian(f,m,z)
        np.testing.assert_allclose(r,a['output_probe_log_resistivity'][0],rtol=2e-12,atol=2e-12)
        np.testing.assert_allclose(p,a['output_probe_phase'][0],rtol=2e-12,atol=2e-12)
        np.testing.assert_allclose(j1,a['output_probe_j1'],rtol=2e-10,atol=2e-12)
        np.testing.assert_allclose(j2,a['output_probe_j2'],rtol=2e-10,atol=2e-12)
        np.testing.assert_allclose(c,a['output_probe_transfer'],rtol=2e-12,atol=2e-9)

def test_independent_truncated_iteration_parity():
    d=load_data(ROOT/'data/raw_data.npz')
    with np.load(ROOT/'test/fixtures/upstream_corrected.npz') as a:
        for k in range(1,7):
            model,h=reconstruct(d,target_rms=0,max_iterations=k)
            np.testing.assert_allclose(model,a['output_truncated_models'][k-1],rtol=2e-9,atol=2e-10)
            np.testing.assert_allclose(h['final_rms'],a['output_truncated_rms'][k-1,0],rtol=2e-9)

def test_full_pipeline_parity():
    model,h=reconstruct(load_data(ROOT/'data/raw_data.npz'))
    with np.load(ROOT/'test/fixtures/upstream_corrected.npz') as a:
        np.testing.assert_allclose(model,a['output_standard_model'][0],rtol=2e-9,atol=2e-10)
    assert h['final_rms']<1.0
    assert h['rms'][2]<h['rms'][1]<h['rms'][0]
