from pathlib import Path
import numpy as np
from src.preprocessing import prepare
from src.physics_model import trajectory
from src.solvers import initial_guess,reconstruct
ROOT=Path(__file__).resolve().parents[2]

def test_original_preprocessing_and_initial_forward():
    data=prepare(ROOT/'data/raw_data.npz')
    with np.load(ROOT/'test/fixtures/upstream.npz') as f:
        assert abs(data['b_minus_a']-float(f['b_minus_a']))<1e-15
        assert abs(data['mu0']-float(f['mu0']))<1e-15
        np.testing.assert_allclose(initial_guess(data),f['initial_log_parameters'],rtol=0,atol=1e-12)
        initial=trajectory(f['initial_log_parameters'],data)
        np.testing.assert_allclose(initial['friction'][0],f['initial_prediction'],atol=1e-5,rtol=0)
        assert np.all(initial['state']>0) and np.all(initial['slider_velocity']>0)

def test_full_bound_original_inverse():
    data=prepare(ROOT/'data/raw_data.npz');out,info=reconstruct(data)
    with np.load(ROOT/'test/fixtures/upstream.npz') as f:
        np.testing.assert_allclose(out['reconstruction'][0,:2],f['reconstruction'][0,:2],atol=2e-4,rtol=0)
        np.testing.assert_allclose(out['reconstruction'][0,2],f['reconstruction'][0,2],atol=5e-8,rtol=0)
        np.testing.assert_allclose(out['state'],f['state'],atol=.05,rtol=0)
        assert np.linalg.norm(out['slider_velocity']-f['slider_velocity'])/np.linalg.norm(f['slider_velocity'])<=.005
        np.testing.assert_allclose(out['friction'],f['friction'],atol=1e-5,rtol=0)
        assert np.sqrt(np.mean((out['friction'][0]-data['friction'])**2))<=.0009888476383917916
        assert info['success']
