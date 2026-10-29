from pathlib import Path
import numpy as np
from src.preprocessing import load_observations
from src.solvers import profile_batch,reconstruct

ROOT=Path(__file__).resolve().parents[2]

def test_independent_supplement_matches_observations_and_varve_measurements():
    with np.load(ROOT/'test/fixtures/upstream_stages.npz') as fixture:
        table=fixture['independent_varve_table']
    for actual,column in zip(load_observations(ROOT/'data/raw_data.npz'),[1,2,3]):
        np.testing.assert_array_equal(actual,table[:,column])
    with np.load(ROOT/'evaluation/ground_truth.npz') as truth:
        np.testing.assert_array_equal(truth['reconstruction'][0],table[:,4])
        np.testing.assert_array_equal(truth['age_uncertainty'][0],table[:,5])

def test_101_independent_author_profiles_and_chi():
    with np.load(ROOT/'test/fixtures/upstream_stages.npz') as fixture:
        chi,profiles=profile_batch(fixture['parameters'],*load_observations(ROOT/'data/raw_data.npz'),fixture['z1'],fixture['z2'])
        np.testing.assert_allclose(chi,fixture['chi'],rtol=1e-9,atol=1e-10)
        np.testing.assert_allclose(profiles,fixture['profiles'],rtol=1e-9,atol=1e-9)

def test_native_full_grid_and_all_confidence_groups():
    profile,details=reconstruct(*load_observations(ROOT/'data/raw_data.npz'))
    assert details['grid'].shape==(810000,4)
    assert [len(c) for c in details['cloud_parameters']]==[481,7864,25064]
    np.testing.assert_allclose(details['minimum_chi'],.584293879254715,rtol=1e-9)
    with np.load(ROOT/'notebooks/reference_outputs/upstream_profiles.npz') as f:
        np.testing.assert_allclose(profile,f['profile'],rtol=1e-9,atol=1e-9)
