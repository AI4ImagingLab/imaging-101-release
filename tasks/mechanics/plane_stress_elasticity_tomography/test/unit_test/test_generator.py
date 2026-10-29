from pathlib import Path
from dataclasses import replace
import numpy as np
from src.generate_data import add_noise,generate_observations
from src.preprocessing import load_data

ROOT=Path(__file__).resolve().parents[2]

def test_sequential_noise_uses_absolute_relative_amplitude_and_post_first_range():
    clean=np.linspace(-2.,4.,882)
    first=clean+.001*np.abs(clean)
    expected=first+.001*np.ptp(first)
    np.testing.assert_array_equal(add_noise(clean,np.ones((2,882))),expected)

def test_noise_statistical_scale_without_cross_library_seed_matching():
    rng=np.random.default_rng(12);clean=np.linspace(.001,.004,882)
    relative=[];global_term=[]
    for sample in range(200):
        draws=rng.standard_normal((2,882));draws[1]=0
        relative.append((add_noise(clean,draws)-clean)/(.001*np.abs(clean)))
        draws=rng.standard_normal((2,882));draws[0]=0
        global_term.append((add_noise(clean,draws)-clean)/(.001*np.ptp(clean)))
    for values in (np.asarray(relative),np.asarray(global_term)):
        assert abs(values.mean())<.02
        assert abs(values.std()-1)<.02

def test_generator_explicit_physical_truth_and_zero_draws():
    raw=load_data(ROOT/'data/raw_data.npz')
    simulated,truth,clean,draws=generate_observations(raw,np.zeros((2,882)))
    np.testing.assert_array_equal(simulated.displacement.ravel(),clean)
    with np.load(ROOT/'test/fixtures/upstream_initial_stages.npz',allow_pickle=False) as source:
        np.testing.assert_array_equal(truth,source['input_modulus'])
        assert np.linalg.norm(clean-source['output_displacement'])/np.linalg.norm(source['output_displacement'])<1e-10

def test_original_noise_realization_with_fresh_local_FEM_never_copies_input_displacement():
    raw=load_data(ROOT/'data/raw_data.npz')
    with np.load(ROOT/'test/fixtures/upstream_noise_draws.npz',allow_pickle=False) as source:
        draws=source['input_standard_normals']
    actual,truth,clean,_=generate_observations(raw,draws)
    assert np.linalg.norm(actual.displacement-raw.displacement)/np.linalg.norm(raw.displacement)<1e-10
    corrupted=replace(raw,displacement=np.full_like(raw.displacement,1e9))
    repeated,*_=generate_observations(corrupted,draws)
    np.testing.assert_array_equal(actual.displacement,repeated.displacement)
