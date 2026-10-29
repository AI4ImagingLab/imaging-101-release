from pathlib import Path
import numpy as np
from src.preprocessing import load_data,observation_weights,element_incenters
from src.physics_model import PlaneStressModel

ROOT=Path(__file__).resolve().parents[2]

def test_original_FEM_on_bound_observations():
    model=PlaneStressModel(load_data(ROOT/'data/raw_data.npz'))
    with np.load(ROOT/'test/fixtures/upstream_initial_stages.npz',allow_pickle=False) as f:
        K=model.assemble(f['input_modulus']).toarray()
        assert np.linalg.norm(K-f['output_stiffness'])/np.linalg.norm(f['output_stiffness'])<1e-12
        predicted=model.forward(f['input_modulus'])
        assert np.linalg.norm(predicted-f['output_displacement'])/np.linalg.norm(f['output_displacement'])<1e-10

def test_source_incenters_and_observation_weights():
    raw=load_data(ROOT/'data/raw_data.npz')
    with np.load(ROOT/'test/fixtures/upstream_initial_stages.npz',allow_pickle=False) as f:
        np.testing.assert_allclose(element_incenters(raw.nodes,raw.triangles),f['output_centers'],rtol=0,atol=2e-14)
        np.testing.assert_allclose(observation_weights(raw.displacement),f['output_weights'],rtol=2e-14,atol=0)
