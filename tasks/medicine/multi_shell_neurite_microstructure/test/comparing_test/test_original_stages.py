"""Source captures fixed before any cleaned inverse mathematics."""
from pathlib import Path
import numpy as np
from numpy.testing import assert_allclose
from src.physics_model import spherical_mean,signal_objective,transform_parameters
from src.solvers import fit_signals
ROOT=Path(__file__).resolve().parents[2]

def test_source_mean_signal_and_independent_orientation_integral():
    with np.load(ROOT/'test/fixtures/original_physics.npz',allow_pickle=False) as f:
        a=f['original_signal_cases'];expected=f['independent_orientation_integral']
    assert_allclose(spherical_mean(a[:,0],a[:,1],a[:,2]),a[:,3],rtol=1e-12,atol=1e-14)
    assert_allclose(a[:,3],expected,rtol=1e-12,atol=1e-14)

def test_initial_objective_and_first_ten_original_updates():
    with np.load(ROOT/'test/fixtures/original_stages.npz',allow_pickle=False) as f:
        y=f['observed'];b=f['bvalues'];cases=f['cases'];steps=f['steps']
    assert_allclose(signal_objective(np.zeros((len(y),2)),y,b),cases[:,-1],rtol=1e-12,atol=1e-8)
    result,diag=fit_signals(y,b,max_iter=10,return_history=True)
    for case in range(len(y)):
        source=steps[steps[:,0]==case]
        assert_allclose(diag['parameter_history'][:,case],source[:,2:4],rtol=1e-10,atol=1e-10)
        assert_allclose(diag['objective_history'][:,case],source[:,4],rtol=1e-10,atol=1e-8)

def test_independent_original_full_float64_sample_fits():
    with np.load(ROOT/'test/fixtures/original_stages.npz',allow_pickle=False) as f:y=f['observed'];b=f['bvalues'];final=f['final']
    fitted,diag=fit_signals(y,b)
    # Source float64 fixtures are stricter than full-field float32 scoring.
    assert_allclose(fitted[:,0],final[:,1],rtol=1e-5,atol=1e-6)
    assert_allclose(fitted[:,1],final[:,2],rtol=1e-5,atol=1e-9)
