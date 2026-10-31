from pathlib import Path
import numpy as np
from src.preprocessing import load_observations, sample_stack
from src.physics_model import triangular_weights
from src.solvers import fit_response, merge_exposures, reconstruct
T=Path(__file__).resolve().parents[2]

def test_upstream_sampling_and_weights():
    frames,times=load_observations(T/'data/raw_data.npz')
    with np.load(T/'test/fixtures/upstream_stages.npz',allow_pickle=False) as f:
        np.testing.assert_array_equal(sample_stack(frames,times),f['output_samples'])
        np.testing.assert_allclose(triangular_weights(np.arange(256)/255),f['output_weights'][0],atol=2e-15,rtol=0)

def test_upstream_response():
    with np.load(T/'test/fixtures/upstream_stages.npz',allow_pickle=False) as f:
        g=fit_response(f['output_samples'],f['input_exposure_times'])
        np.testing.assert_allclose(g,f['output_g'],atol=2e-9,rtol=2e-9)

def test_upstream_merging_and_full_pipeline():
    frames,times=load_observations(T/'data/raw_data.npz')
    with np.load(T/'evaluation/baseline_reference.npz',allow_pickle=False) as f:
        merged=merge_exposures(frames,times,f['response'][0])
        np.testing.assert_allclose(merged,f['reconstruction'][0],rtol=3e-6,atol=2e-6)
        fresh=reconstruct(frames,times)
        np.testing.assert_allclose(fresh['response'],f['response'],rtol=2e-9,atol=2e-9)
        np.testing.assert_allclose(fresh['reconstruction'],f['reconstruction'],rtol=3e-6,atol=2e-6)
