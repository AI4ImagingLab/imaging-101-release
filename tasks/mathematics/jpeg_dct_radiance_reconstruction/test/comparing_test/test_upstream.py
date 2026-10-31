from pathlib import Path
import numpy as np
from src.physics_model import inverse,tv_gradient,objective,transform
from src.preprocessing import load_observations
from src.solvers import reconstruct
ROOT=Path(__file__).resolve().parents[2]
def test_independently_captured_original_decode_and_gradient():
    a,q=load_observations(ROOT/'data/raw_data.npz')
    with np.load(ROOT/'test/fixtures/upstream_states.npz') as z:
        u=z['output_decode']
        np.testing.assert_allclose(inverse(a*q.astype(np.float32)[:,None,None]),u,rtol=0,atol=.0001)
        np.testing.assert_allclose(tv_gradient(u),z['output_initial_tv_gradient'],rtol=0,atol=.000002)
def test_native_full_pipeline_and_objective_parity():
    a,q=load_observations(ROOT/'data/raw_data.npz');u=reconstruct(a,q)
    with np.load(ROOT/'test/fixtures/upstream_states.npz') as z:ref=z['output_iterations_2000']
    assert np.sqrt(np.mean((u-ref).astype(float)**2))/255<.00005
    assert abs(objective(u,a,q)/objective(ref,a,q)-1)<.00002
    assert np.max(np.abs(transform(u)-a*q[:,None,None])-.5*q[:,None,None])<.001
    with np.load(ROOT/'output/reconstruction.npz') as z:submission=z['reconstruction']
    np.testing.assert_array_equal(submission,u[None])
