from pathlib import Path
import numpy as np
from src.physics_model import resonances
from src.solvers import reconstruct
ROOT=Path(__file__).resolve().parents[2]

def test_frozen_original_forward():
    with np.load(ROOT/'test/fixtures/upstream.npz') as f:
        np.testing.assert_allclose(resonances([2.7,.8]),f['nominal'],rtol=2e-6,atol=3e-6)
        np.testing.assert_allclose(resonances([2.8,.85]),f['perturbed'],rtol=2e-6,atol=3e-6)

def test_complete_inverse_and_independent_prior_stability():
    with np.load(ROOT/'test/fixtures/upstream.npz') as f:
        for prior in [(2.5,.7),(2.6,.75),(2.7,.8),(2.8,.85),(3.,.9),(3.2,.95)]:
            output,details=reconstruct(f['measured'],f['weights'],prior)
            np.testing.assert_array_equal(details['mode_indices'],f['manual_mode_indices'])
            np.testing.assert_allclose(output['reconstruction'][0],f['moduli'],rtol=0,atol=.002)
            np.testing.assert_allclose(output['resonances'][0],f['native_frequencies'],rtol=0,atol=3e-6)
            assert details['weighted_rms_percent']<.08523581
